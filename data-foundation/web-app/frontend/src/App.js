import React, { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import DataManagement from './components/DataManagement';
import Analytics from './components/Analytics';

function App() {
  const [activeView, setActiveView] = useState('data-management');

  const renderContent = () => {
    switch (activeView) {
      case 'data-management':
        return <DataManagement />;
      case 'analytics':
        return <Analytics />;
      default:
        return <DataManagement />;
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>EHS AI Assistant</h1>
      </header>
      
      <div className="main-content">
        <Sidebar activeView={activeView} onViewChange={setActiveView} />
        <main className="content">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

export default App;