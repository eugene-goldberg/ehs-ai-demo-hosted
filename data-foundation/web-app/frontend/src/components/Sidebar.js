import React from 'react';
import './Sidebar.css';

const Sidebar = ({ activeView, onViewChange }) => {
  const menuItems = [
    {
      id: 'data-management',
      label: 'Data Management',
      icon: '📁',
      count: 24
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: '📊',
      count: 8
    },
    {
      id: 'analytics-hierarchical',
      label: 'Analytics (Hierarchical)',
      icon: '🏢',
      count: null
    },
    {
      id: 'analytics-executive',
      label: 'Analytics (Executive)',
      icon: '👔',
      count: null
    }
  ];

  return (
    <div className="sidebar">
      <nav className="sidebar-nav">
        <ul className="nav-list">
          {menuItems.map((item) => (
            <li key={item.id} className="nav-item">
              <button
                className={`nav-link ${activeView === item.id ? 'active' : ''}`}
                onClick={() => onViewChange(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                {item.count && (
                  <span className="nav-count">{item.count}</span>
                )}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;