#!/usr/bin/env python3
"""
Create sample EHS data for testing vector and fulltext indexes
"""

import os
import logging
import uuid
from datetime import datetime
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EHSSampleDataCreator:
    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # Initialize connections
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize OpenAI for embeddings
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key required")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_embedding(self, text):
        """Get OpenAI embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def create_sample_documents(self):
        """Create sample EHS documents with content"""
        logger.info("Creating sample EHS documents...")
        
        sample_documents = [
            {
                'title': 'Environmental Compliance Manual',
                'content': '''This environmental compliance manual outlines the requirements for maintaining environmental 
                compliance across all facilities. It covers air quality monitoring, water discharge permits, 
                waste management procedures, and regulatory reporting requirements. All personnel must follow 
                these environmental protection protocols to ensure compliance with EPA regulations and 
                state environmental laws. Regular environmental audits are conducted to verify compliance.''',
                'document_type': 'compliance_manual',
                'summary': 'Comprehensive guide for environmental compliance requirements and procedures'
            },
            {
                'title': 'Air Quality Monitoring Report Q3 2024',
                'content': '''Quarterly air quality monitoring report for Q3 2024. This report includes emissions 
                data from all stack monitoring systems, ambient air quality measurements, and compliance 
                status with Clean Air Act requirements. Particulate matter (PM2.5 and PM10) levels were 
                within acceptable limits. NOx emissions showed a 5% reduction compared to Q2. All continuous 
                emission monitoring systems are operating properly and calibrated according to EPA standards.''',
                'document_type': 'monitoring_report',
                'summary': 'Q3 2024 air quality monitoring results and emissions compliance status'
            },
            {
                'title': 'Hazardous Waste Management Procedures',
                'content': '''This document establishes procedures for the safe handling, storage, and disposal 
                of hazardous waste materials. All hazardous waste must be properly classified according to 
                RCRA standards and stored in appropriate containers. Waste manifests must be completed for 
                all shipments to licensed disposal facilities. Emergency response procedures are included 
                for spill containment and cleanup. Training requirements for hazardous waste handlers are outlined.''',
                'document_type': 'procedure',
                'summary': 'Procedures for safe hazardous waste handling and disposal'
            },
            {
                'title': 'Facility Environmental Permit Review',
                'content': '''Annual review of environmental permits for Manufacturing Facility A. Current permits 
                include air emissions permit, NPDES water discharge permit, and hazardous waste storage permit. 
                All permits are current and in good standing. Permit renewal dates are tracked and renewal 
                applications will be submitted 180 days prior to expiration. Recent permit modifications 
                include updated emission limits for the new production line.''',
                'document_type': 'permit_review',
                'summary': 'Annual environmental permit status review for Manufacturing Facility A'
            },
            {
                'title': 'Chemical Safety Data Sheet - Methylene Chloride',
                'content': '''Safety data sheet for methylene chloride (dichloromethane) used in manufacturing 
                processes. This chemical requires special handling procedures due to its carcinogenic properties. 
                Personal protective equipment includes respiratory protection, chemical-resistant gloves, and 
                eye protection. Storage requirements include ventilated areas away from heat sources. 
                Emergency procedures for exposure include immediate medical attention and decontamination protocols.''',
                'document_type': 'safety_data_sheet',
                'summary': 'Safety information and handling procedures for methylene chloride'
            }
        ]
        
        created_docs = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            for doc_data in sample_documents:
                doc_id = str(uuid.uuid4())
                
                # Get embeddings
                title_embedding = self.get_embedding(doc_data['title'])
                content_embedding = self.get_embedding(doc_data['content'])
                summary_embedding = self.get_embedding(doc_data['summary'])
                
                # Create document
                query = """
                CREATE (d:Document {
                    id: $doc_id,
                    title: $title,
                    content: $content,
                    summary: $summary,
                    document_type: $document_type,
                    created_date: $created_date,
                    title_embedding: $title_embedding,
                    content_embedding: $content_embedding,
                    summary_embedding: $summary_embedding
                })
                RETURN d.id as id
                """
                
                result = session.run(query, {
                    'doc_id': doc_id,
                    'title': doc_data['title'],
                    'content': doc_data['content'],
                    'summary': doc_data['summary'],
                    'document_type': doc_data['document_type'],
                    'created_date': datetime.now().isoformat(),
                    'title_embedding': title_embedding,
                    'content_embedding': content_embedding,
                    'summary_embedding': summary_embedding
                })
                
                created_doc_id = result.single()['id']
                created_docs.append(created_doc_id)
                logger.info(f"Created document: {doc_data['title']}")
        
        logger.info(f"Created {len(created_docs)} sample documents")
        return created_docs
    
    def create_sample_chunks(self, document_ids):
        """Create sample document chunks"""
        logger.info("Creating sample document chunks...")
        
        sample_chunks = [
            {
                'content': '''Environmental compliance requires regular monitoring of air emissions from all 
                stack sources. Continuous emission monitoring systems must be installed and maintained 
                according to EPA specifications. Data must be reported quarterly to regulatory agencies.''',
                'section_title': 'Air Emissions Monitoring',
                'chunk_index': 1
            },
            {
                'content': '''Water discharge permits must be renewed every five years. All discharge points 
                require monthly sampling and analysis. Results must be reported to the state environmental 
                agency within 30 days of sampling. Exceedances must be reported immediately.''',
                'section_title': 'Water Discharge Compliance',
                'chunk_index': 2
            },
            {
                'content': '''Hazardous waste storage areas must be inspected weekly for container integrity, 
                proper labeling, and secondary containment. Any deficiencies must be corrected immediately. 
                Inspection records must be maintained for at least three years.''',
                'section_title': 'Hazardous Waste Storage',
                'chunk_index': 3
            },
            {
                'content': '''Personal protective equipment is required when handling chemicals with health 
                hazards. Respirators must be fit-tested annually. Eye wash stations and safety showers 
                must be tested monthly. All safety equipment inspections must be documented.''',
                'section_title': 'Chemical Safety Procedures',
                'chunk_index': 4
            },
            {
                'content': '''Emergency response procedures must be reviewed annually and updated as needed. 
                All employees must receive emergency response training. Spill response equipment must be 
                readily available and inspected monthly. Emergency contact information must be posted.''',
                'section_title': 'Emergency Response',
                'chunk_index': 5
            }
        ]
        
        created_chunks = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            for i, chunk_data in enumerate(sample_chunks):
                chunk_id = str(uuid.uuid4())
                
                # Get embedding
                content_embedding = self.get_embedding(chunk_data['content'])
                
                # Use the first document for all chunks (could be randomized)
                doc_id = document_ids[0] if document_ids else None
                
                # Create chunk
                query = """
                CREATE (c:DocumentChunk {
                    id: $chunk_id,
                    content: $content,
                    section_title: $section_title,
                    chunk_index: $chunk_index,
                    content_embedding: $content_embedding,
                    created_date: $created_date
                })
                """
                
                # If we have a document, create relationship
                if doc_id:
                    query += """
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    CREATE (d)-[:HAS_CHUNK]->(c)
                    """
                
                query += " RETURN c.id as id"
                
                result = session.run(query, {
                    'chunk_id': chunk_id,
                    'content': chunk_data['content'],
                    'section_title': chunk_data['section_title'],
                    'chunk_index': chunk_data['chunk_index'],
                    'content_embedding': content_embedding,
                    'created_date': datetime.now().isoformat(),
                    'doc_id': doc_id
                })
                
                created_chunk_id = result.single()['id']
                created_chunks.append(created_chunk_id)
                logger.info(f"Created chunk: {chunk_data['section_title']}")
        
        logger.info(f"Created {len(created_chunks)} sample chunks")
        return created_chunks
    
    def create_sample_facilities(self):
        """Create sample facilities with descriptions"""
        logger.info("Creating sample facilities...")
        
        sample_facilities = [
            {
                'name': 'Manufacturing Facility A',
                'description': '''Large manufacturing facility specializing in chemical processing and 
                production. The facility operates under multiple environmental permits including air 
                emissions, water discharge, and hazardous waste storage. Regular environmental monitoring 
                ensures compliance with all applicable regulations.''',
                'facility_type': 'manufacturing',
                'location': 'Industrial District, State A'
            },
            {
                'name': 'Distribution Center B',
                'description': '''Regional distribution center for finished products. Facility includes 
                warehouse storage, loading docks, and administrative offices. Environmental considerations 
                include stormwater management, waste minimization, and energy efficiency programs.''',
                'facility_type': 'distribution',
                'location': 'Commercial Zone, State B'
            }
        ]
        
        created_facilities = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            for facility_data in sample_facilities:
                facility_id = str(uuid.uuid4())
                
                # Get embedding
                description_embedding = self.get_embedding(facility_data['description'])
                
                # Create facility
                query = """
                CREATE (f:Facility {
                    id: $facility_id,
                    name: $name,
                    description: $description,
                    facility_type: $facility_type,
                    location: $location,
                    description_embedding: $description_embedding,
                    created_date: $created_date
                })
                RETURN f.id as id
                """
                
                result = session.run(query, {
                    'facility_id': facility_id,
                    'name': facility_data['name'],
                    'description': facility_data['description'],
                    'facility_type': facility_data['facility_type'],
                    'location': facility_data['location'],
                    'description_embedding': description_embedding,
                    'created_date': datetime.now().isoformat()
                })
                
                created_facility_id = result.single()['id']
                created_facilities.append(created_facility_id)
                logger.info(f"Created facility: {facility_data['name']}")
        
        logger.info(f"Created {len(created_facilities)} sample facilities")
        return created_facilities
    
    def create_all_sample_data(self):
        """Create all sample data"""
        logger.info("Creating comprehensive EHS sample data...")
        
        # Create documents
        document_ids = self.create_sample_documents()
        
        # Create chunks
        chunk_ids = self.create_sample_chunks(document_ids)
        
        # Create facilities
        facility_ids = self.create_sample_facilities()
        
        logger.info("=" * 60)
        logger.info("SAMPLE DATA CREATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Documents created: {len(document_ids)}")
        logger.info(f"Chunks created: {len(chunk_ids)}")
        logger.info(f"Facilities created: {len(facility_ids)}")
        logger.info("=" * 60)
        
        return {
            'documents': document_ids,
            'chunks': chunk_ids,
            'facilities': facility_ids
        }

def main():
    """Main function"""
    creator = EHSSampleDataCreator()
    
    try:
        results = creator.create_all_sample_data()
        logger.info("🎉 Sample data creation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Sample data creation failed: {e}")
        return 1
    finally:
        creator.close()

if __name__ == "__main__":
    exit(main())
