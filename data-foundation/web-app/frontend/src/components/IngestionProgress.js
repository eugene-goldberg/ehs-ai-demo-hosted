import React, { useState, useEffect } from 'react';
import './IngestionProgress.css';

const IngestionProgress = ({ isIngesting }) => {
  const stages = [
    'validate',
    'parse',
    'extract',
    'transform',
    'validate_data',
    'load',
    'index',
    'complete'
  ];

  const [currentStage, setCurrentStage] = useState(0);
  const [completedStages, setCompletedStages] = useState([]);

  useEffect(() => {
    if (!isIngesting) {
      setCurrentStage(0);
      setCompletedStages([]);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev < stages.length - 1) {
          setCompletedStages((completed) => [...completed, stages[prev]]);
          return prev + 1;
        }
        return prev;
      });
    }, 11000); // ~90 seconds total / 8 stages

    return () => clearInterval(interval);
  }, [isIngesting]);

  if (!isIngesting) return null;

  return (
    <div className="ingestion-progress">
      <div className="progress-stages">
        {stages.map((stage, index) => (
          <div
            key={stage}
            className={`stage ${
              completedStages.includes(stage) ? 'completed' : ''
            } ${currentStage === index ? 'active' : ''}`}
          >
            <span className="stage-icon">
              {completedStages.includes(stage) ? '✓' : 
               currentStage === index ? '⟳' : '○'}
            </span>
            <span className="stage-name">{stage.replace('_', ' ')}</span>
          </div>
        ))}
      </div>
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${(currentStage / (stages.length - 1)) * 100}%` }}
        />
      </div>
    </div>
  );
};

export default IngestionProgress;