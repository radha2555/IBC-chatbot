import React, { useState, useEffect, useRef } from 'react';

const IBChatbot = () => {
  // State variables
  const [question, setQuestion] = useState('');
  const [dataAnswer, setDataAnswer] = useState({ acts: '', rules: '' });
  const [caseLaws, setCaseLaws] = useState({
    supreme_court: [],
    high_court: [],
    nclat: [],
    nclt: []
  });
  const [selectedCase, setSelectedCase] = useState(null);
  const [clipboardContent, setClipboardContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('acts');
  const [activeCaseTab, setActiveCaseTab] = useState('supreme_court');
  const [apiStatus, setApiStatus] = useState('connecting');
  const [completedSearches, setCompletedSearches] = useState({
    acts: false,
    rules: false,
    supreme_court: false,
    high_court: false,
    nclat: false,
    nclt: false
  });
  
  // Refs
  const answerBoxRef = useRef(null);
  const caseDetailsRef = useRef(null);
  const caseListRef = useRef(null);

  // API Configuration
  const API_BASE_URL = 'http://localhost:8000';

  // Check backend connection on component mount
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/collections`);
        if (response.ok) {
          setApiStatus('connected');
        } else {
          setApiStatus('error');
        }
      } catch (error) {
        console.error('Backend connection error:', error);
        setApiStatus('error');
      }
    };
    checkApiConnection();
  }, []);

  // Generic fetch function for a single collection
  const fetchCollectionData = async (collection, displayName) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, collection, k: 3 })
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch ${displayName} data`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching ${displayName}:`, error);
      if (collection === 'acts' || collection === 'rules') {
        return { answer: `Error loading ${displayName} data` };
      }
      return { sources: [] };
    }
  };

  // Handle form submission with new sequential logic
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || apiStatus !== 'connected') return;

    setIsLoading(true);
    setDataAnswer({ acts: '', rules: '' });
    setCaseLaws({ supreme_court: [], high_court: [], nclat: [], nclt: [] });
    setSelectedCase(null);
    setClipboardContent('');
    setCompletedSearches({ acts: false, rules: false, supreme_court: false, high_court: false, nclat: false, nclt: false });

    // Fetch Acts and Rules independently
    const fetchActs = async () => {
      try {
        const actsResult = await fetchCollectionData('acts', 'Acts');
        setDataAnswer(prev => ({ ...prev, acts: actsResult.answer }));
      } catch (error) {
        console.error("Error fetching Acts:", error);
        setDataAnswer(prev => ({ ...prev, acts: 'An unexpected error occurred while fetching Acts.' }));
      } finally {
        setCompletedSearches(prev => ({ ...prev, acts: true }));
      }
    };

    const fetchRules = async () => {
      try {
        const rulesResult = await fetchCollectionData('rules', 'Rules');
        setDataAnswer(prev => ({ ...prev, rules: rulesResult.answer }));
      } catch (error) {
        console.error("Error fetching Rules:", error);
        setDataAnswer(prev => ({ ...prev, rules: 'An unexpected error occurred while fetching Rules.' }));
      } finally {
        setCompletedSearches(prev => ({ ...prev, rules: true }));
      }
    };

    // Start both fetches without waiting for each other
    fetchActs();
    fetchRules();

    // Fetch case laws sequentially
    const caseCollections = [
      { name: 'supreme_court', displayName: 'Supreme Court' },
      { name: 'high_court', displayName: 'High Court' },
      { name: 'nclat', displayName: 'NCLAT' },
      { name: 'nclt', displayName: 'NCLT' }
    ];

    for (const collection of caseCollections) {
      const caseResult = await fetchCollectionData(collection.name, collection.displayName);
      
      setCaseLaws(prev => ({
        ...prev,
        [collection.name]: caseResult.sources || []
      }));
      setCompletedSearches(prev => ({ ...prev, [collection.name]: true }));
    }

    setIsLoading(false);
  };

  // Rest of the component remains the same...
  // Case selection handler
  const handleCaseClick = (caseLaw) => {
    setSelectedCase(caseLaw);
  };

  // Add content to clipboard
  const handleAddToClipboard = (content) => {
    if (content) {
      setClipboardContent(prev => prev + (prev ? '\n\n' : '') + content);
    }
  };

  // Reset all state
  const handleRefresh = () => {
    if (clipboardContent) downloadClipboard();
    setQuestion('');
    setDataAnswer({ acts: '', rules: '' });
    setCaseLaws({ supreme_court: [], high_court: [], nclat: [], nclt: [] });
    setSelectedCase(null);
    setClipboardContent('');
    setCompletedSearches({ acts: false, rules: false, supreme_court: false, high_court: false, nclat: false, nclt: false });
  };

  // Copy text to system clipboard
  const copyToClipboard = (text) => {
     try {
      navigator.clipboard.writeText(text);
      alert('Copied to clipboard!');
    } catch (err) {
      console.error('Copy failed:', err);
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position="fixed";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
        alert('Copied to clipboard!');
      } catch (err) {
        alert('Failed to copy text.');
      }
      document.body.removeChild(textArea);
    }
  };

  // Download clipboard content
  const downloadClipboard = () => {
    const blob = new Blob([clipboardContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `IBChatbot-Notes-${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Helper functions for copying specific content
  const copyAnswerToClipboard = () => copyToClipboard(dataAnswer[activeTab]);
  const copyCaseDetailsToClipboard = () => {
    if (selectedCase) {
      copyToClipboard(
        `Case: ${selectedCase.metadata.case_name || 'N/A'}\n` +
        `Summary: ${selectedCase.page_content.substring(0, 200)}...`
      );
    }
  };

  const copyAllCasesToClipboard = () => {
    const allCasesText = Object.entries(caseLaws).flatMap(([courtType, cases]) => {
      return cases.length === 0 ? [] : cases.map((caseLaw, index) => 
        `${courtType.replace('_', ' ').toUpperCase()} Case ${index + 1}:\n` +
        `Name: ${caseLaw.metadata.case_name || 'N/A'}\n` +
        `Summary: ${caseLaw.page_content.substring(0, 150)}...`
      );
    }).join('\n\n');
    
    allCasesText ? copyToClipboard(allCasesText) : alert('No cases to copy');
  };

  // Auto-scroll behaviors
  useEffect(() => {
    if (answerBoxRef.current) {
      answerBoxRef.current.scrollTop = answerBoxRef.current.scrollHeight;
    }
  }, [dataAnswer]);

  useEffect(() => {
    if (caseDetailsRef.current && selectedCase) {
      caseDetailsRef.current.scrollTop = 0;
    }
  }, [selectedCase]);

  // Get all cases combined
  const getAllCases = () => {
    return Object.entries(caseLaws).flatMap(([courtType, cases]) => {
      return cases.map(caseLaw => ({
        ...caseLaw,
        courtType: courtType.replace('_', ' ').toUpperCase()
      }));
    });
  };

  const allCases = getAllCases();
  const activeCourtCases = caseLaws[activeCaseTab] || [];

  // Status indicators for each search
  const getSearchStatus = (collection) => {
    if (isLoading && !completedSearches[collection]) return 'Searching...';
    if (!completedSearches[collection]) return 'Pending';
    
    if (collection === 'acts' || collection === 'rules') {
       return dataAnswer[collection] && !dataAnswer[collection].startsWith('Error') ? 'Found' : 'Not found';
    }
    return caseLaws[collection]?.length > 0 ? `Found (${caseLaws[collection].length})` : 'Not found';
  };

  const css = `
    :root {
      --primary-bg: #f4f7f9;
      --secondary-bg: #ffffff;
      --border-color: #e0e5ea;
      --text-primary: #1c2a38;
      --text-secondary: #5a6b7b;
      --accent-color: #007bff;
      --accent-hover: #0056b3;
      --status-connected: #28a745;
      --status-error: #dc3545;
      --status-pending: #ffc107;
      --box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .ibc-chatbot-container {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
      width: 100%;
      background-color: var(--primary-bg);
      color: var(--text-primary);
      padding: 1rem;
      box-sizing: border-box;
    }
    .connection-status {
      position: absolute;
      top: 0;
      left: 0;
      padding: 0.25rem 0.5rem;
      font-size: 0.75rem;
      border-bottom-right-radius: 0.5rem;
    }
    .connection-status.connected { background-color: var(--status-connected); color: white; }
    .connection-status.error { background-color: var(--status-error); color: white; }
    .connection-status.connecting { background-color: var(--status-pending); color: black; }
    
    .top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      padding: 0 0.5rem;
    }
    .logo { font-weight: bold; font-size: 1.5rem; }
    .title { font-size: 1.5rem; font-weight: 500; margin: 0; }
    
    .search-status-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      background-color: var(--secondary-bg);
      padding: 0.5rem;
      border-radius: 8px;
      margin-bottom: 1rem;
      font-size: 0.8rem;
    }
    .status-item {
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      background-color: #e9ecef;
      color: var(--text-secondary);
      transition: all 0.3s ease;
    }
     .status-item.completed { background-color: #d4edda; color: #155724; }
    
    .main-content {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      grid-template-rows: auto;
      gap: 1rem;
      flex-grow: 1;
      overflow: hidden;
    }
    
    .content-box {
      background-color: var(--secondary-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      box-shadow: var(--box-shadow);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .answer-box { grid-column: 1 / 2; }
    .case-law-box { grid-column: 2 / 3; }
    .case-details-box { grid-column: 3 / 4; }
    
    .box-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem;
      border-bottom: 1px solid var(--border-color);
      flex-shrink: 0;
    }
    .box-header h2 { font-size: 1.1rem; margin: 0; }

    .data-tabs, .case-tabs {
      display: flex;
      gap: 0.5rem;
    }

    .tab-button {
      padding: 0.3rem 0.8rem;
      border: 1px solid var(--border-color);
      background-color: transparent;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      transition: all 0.2s ease;
    }
    .tab-button.active { background-color: var(--accent-color); color: white; border-color: var(--accent-color); }
    .tab-button:not(.active):hover { background-color: #e9ecef; }
    
    .scrollable-content {
      padding: 1rem;
      overflow-y: auto;
      flex-grow: 1;
    }
    .answer-content { white-space: pre-wrap; word-wrap: break-word; }
    
    .case-list { display: flex; flex-direction: column; gap: 0.75rem; }
    .case-item {
      padding: 0.75rem;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .case-item:hover { border-color: var(--accent-color); transform: translateY(-2px); }
    .case-item.active { border-color: var(--accent-color); background-color: #e7f3ff; }
    .case-item h3 { font-size: 0.95rem; margin: 0 0 0.25rem; }
    .case-item p { font-size: 0.8rem; color: var(--text-secondary); margin: 0; }
    
    .case-details h3 { margin-top: 0; }
    
    .clipboard-box {
      margin-top: 1rem;
      display: flex;
      flex-direction: column;
      background: var(--secondary-bg);
      border-radius: 8px;
      box-shadow: var(--box-shadow);
      max-height: 250px;
    }
    .clipboard-actions { display: flex; gap: 0.5rem; }
    .clipboard-content {
      flex-grow: 1;
      width: 100%;
      padding: 0.75rem;
      border: none;
      border-top: 1px solid var(--border-color);
      border-bottom-left-radius: 8px;
      border-bottom-right-radius: 8px;
      resize: none;
      font-family: inherit;
    }
    .clipboard-content:focus { outline: none; }
    
    .chat-input {
      display: flex;
      margin-top: 1rem;
      gap: 0.5rem;
    }
    .chat-input input {
      flex-grow: 1;
      padding: 0.75rem 1rem;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      font-size: 1rem;
    }
    .chat-input input:focus { outline: none; border-color: var(--accent-color); }
    
    button {
      padding: 0.5rem 1rem;
      font-size: 0.9rem;
      border: none;
      border-radius: 6px;
      background-color: var(--accent-color);
      color: white;
      cursor: pointer;
      transition: background-color 0.2s ease;
    }
    button:hover { background-color: var(--accent-hover); }
    button:disabled { background-color: #ced4da; cursor: not-allowed; }
    .copy-button { background-color: #6c757d; }
    .copy-button:hover { background-color: #5a6268; }
    
    .loading-spinner {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
    }
    .empty-state {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
    }
    
    @media (max-width: 900px) {
      .main-content {
        grid-template-columns: 1fr 1fr;
      }
      .answer-box { grid-column: 1 / 2; }
      .case-law-box { grid-column: 2 / 3; }
      .case-details-box { grid-column: 1 / 3; }
    }

    @media (max-width: 600px) {
      .main-content {
        grid-template-columns: 1fr;
        gap: 0.5rem;
      }
      .answer-box, .case-law-box, .case-details-box { grid-column: auto; }
      .top-bar { flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }
      .clipboard-box { max-height: 150px; }
    }
  `;

  return (
    <>
      <style>{css}</style>
      <div className="ibc-chatbot-container">
        <div className={`connection-status ${apiStatus}`}>
          Backend: {apiStatus}
        </div>

        <header className="top-bar">
          <div className="logo">IBC</div>
          <h1 className="title">Legal Research Assistant</h1>
          <button className="refresh-button" onClick={handleRefresh}>
            ↻ Refresh & Save
          </button>
        </header>

        <div className="search-status-bar">
          <div className={`status-item ${completedSearches.acts ? 'completed' : ''}`}>
            Acts: {getSearchStatus('acts')}
          </div>
          <div className={`status-item ${completedSearches.rules ? 'completed' : ''}`}>
            Rules: {getSearchStatus('rules')}
          </div>
          <div className={`status-item ${completedSearches.supreme_court ? 'completed' : ''}`}>
            SC: {getSearchStatus('supreme_court')}
          </div>
          <div className={`status-item ${completedSearches.high_court ? 'completed' : ''}`}>
            HC: {getSearchStatus('high_court')}
          </div>
          <div className={`status-item ${completedSearches.nclat ? 'completed' : ''}`}>
            NCLAT: {getSearchStatus('nclat')}
          </div>
          <div className={`status-item ${completedSearches.nclt ? 'completed' : ''}`}>
            NCLT: {getSearchStatus('nclt')}
          </div>
        </div>

        <div className="main-content">
          <div className="content-box answer-box">
            <div className="box-header">
              <h2> Acts & Rules </h2>
              <div className="data-tabs">
                <button 
                  className={`tab-button ${activeTab === 'acts' ? 'active' : ''}`}
                  onClick={() => setActiveTab('acts')}
                >
                  Acts
                </button>
                <button 
                  className={`tab-button ${activeTab === 'rules' ? 'active' : ''}`}
                  onClick={() => setActiveTab('rules')}
                >
                  Rules
                </button>
              </div>
            </div>
            <div className="scrollable-content" ref={answerBoxRef}>
              {isLoading && !completedSearches[activeTab] ? (
                <div className="loading-spinner">Searching {activeTab}...</div>
              ) : (
                <div className="answer-content">
                  {dataAnswer[activeTab] || `No ${activeTab} data found.`}
                </div>
              )}
            </div>
          </div>

          <div className="content-box case-law-box">
            <div className="box-header">
              <h2> Cases </h2>
              <div className="case-tabs">
                {['supreme_court', 'high_court', 'nclat', 'nclt'].map(court => (
                  <button
                    key={court}
                    className={`tab-button ${activeCaseTab === court ? 'active' : ''}`}
                    onClick={() => setActiveCaseTab(court)}
                  >
                    {court.replace('_', ' ').replace('court', 'C').toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="scrollable-content" ref={caseListRef}>
              <div className="case-list">
                {isLoading && !completedSearches[activeCaseTab] ? (
                  <div className="loading-spinner">Searching {activeCaseTab.replace('_', ' ')}...</div>
                ) : activeCourtCases.length > 0 ? (
                  activeCourtCases.map((caseLaw, index) => (
                    <div 
                      key={caseLaw.id || index} 
                      className={`case-item ${selectedCase === caseLaw ? 'active' : ''}`}
                      onClick={() => handleCaseClick(caseLaw)}
                    >
                      <h3>{caseLaw.metadata.case_name || `Case ${index + 1}`}</h3>
                      <p>{caseLaw.page_content.substring(0, 100)}...</p>
                    </div>
                  ))
                ) : (
                  <p className="empty-state">
                    {completedSearches[activeCaseTab] ? 'No cases found.' : 'Search not started.'}
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className="content-box case-details-box">
            <div className="box-header">
              <h2>Case Details</h2>
              <button 
                className="copy-button" 
                onClick={() => handleAddToClipboard(
                  `Case: ${selectedCase?.metadata.case_name || 'N/A'}\n` +
                  `Summary: ${selectedCase?.page_content || ''}`
                )}
                disabled={!selectedCase}
              >
                Add to Notes
              </button>
            </div>
            <div className="scrollable-content" ref={caseDetailsRef}>
              {selectedCase ? (
                <div className="case-details">
                  <h3>{selectedCase.metadata.case_name || 'Case Details'}</h3>
                  <p>{selectedCase.page_content}</p>
                </div>
              ) : (
                <p className="empty-state">Select a case to view details</p>
              )}
            </div>
          </div>
        </div>

        <div className="clipboard-box">
          <div className="box-header">
            <h2>Notes</h2>
            <div className="clipboard-actions">
              <button onClick={() => copyToClipboard(clipboardContent)} disabled={!clipboardContent}>
                Copy
              </button>
              <button onClick={downloadClipboard} disabled={!clipboardContent}>
                Download
              </button>
              <button onClick={() => setClipboardContent('')} disabled={!clipboardContent}>
                Clear
              </button>
            </div>
          </div>
          <textarea
            className="clipboard-content"
            value={clipboardContent}
            onChange={(e) => setClipboardContent(e.target.value)}
            placeholder="Your notes and saved content will appear here..."
          />
        </div>

        <form className="chat-input" onSubmit={handleSubmit}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Write your questions here..."
            disabled={isLoading || apiStatus !== 'connected'}
          />
          <button 
            type="submit" 
            disabled={isLoading || apiStatus !== 'connected'}
          >
            {isLoading ? 'Searching...' : 'Ask'}
          </button>
        </form>
      </div>
    </>
  );
};

export default IBChatbot;