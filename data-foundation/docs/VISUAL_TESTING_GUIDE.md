Follow these instructions to perform the visual testing of the 'Analytics (Executive)' dashboard:
1   Before proceeding make sure that the ingestion workflow has the 'clear database' flag set to false. This is done in order to prevent the 6 month of historical data from being removed from the Neo4j database
2   Use the playwright MCP tool to conduct the entire round of visual testing
3   Open the Google Chrome (default) browser using the playwrite MCP
4   Open the Developer Tool console in the browser to be able to monitor for any UI errors
4.1 Clock on the 'Rest' button to clear out the 'Processed Docuemnts' table before running the ingestion workflow
5   Click on the 'Ingest Invoices' button under the 'Data Management' section of the UI
6   Observe the visual progress indicator
7   The data ingestion process usually takes between 1.5 to 3 minutes, so you need to keep sleeping for 20 seconds and repeat your screenshots to capture the progress of the visual progress indicator
8 Once the data ingestion finishes you will see the 'Processed Documents' table populated with ingested records
9   Since it is the part of our new functionality to collect all langsmith traces (the ones from the ingestion and the ones from the risk assessment agent), you will click on the 'expend transcript' arrow botton located next to the 'LLM Interaction Transcript' label
10 Within the expended transcipt section you should be able to find the langsmith traces from both the data ingestion as well as the riask assement agent runs
11 Once these traces have been verified, you will click on the 'Analytics (Executove)' navigation link, which will open the dashboard that you need to inspect
12 The dashboard should be populated using the Neo4j risks and recommendations data which was by the risk assessment agent during the last step of the ingestion process. 
13 You must verify that data and compare it with what you expected that data to be
14 You must verify that the risk assessment agent is actually being called as the last setp of ingestion
15 You must Check if the executive dashboard API v2 is integrated with the frontend
16 Ensure LangSmith traces from the risk assessment agent are being captured and displayed along with the data ingestion traces
17 During the data ingestion process you must not ignore the errors which may appear in the browser console, as they indicate that some of your work is broken and you must understand the cause of each error and fox it