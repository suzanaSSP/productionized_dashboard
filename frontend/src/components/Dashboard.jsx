import React, { useState, useEffect } from 'react';
import { getCustomerSegments, getRFMMetrics, getModelMetrics, exportCustomersCSV } from '../services/api';
import '../styles/Dashboard.css';

const Dashboard = () => {
  const [segments, setSegments] = useState([]);
  const [rfmData, setRfmData] = useState([]);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ segment: '' });

  // Fetch all data on component load
  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      const [segmentsData, rfmData, metricsData] = await Promise.all([
        getCustomerSegments(),
        getRFMMetrics(),
        getModelMetrics(),
      ]);
      setSegments(segmentsData);
      setRfmData(rfmData);
      setModelMetrics(metricsData);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters({ ...filters, [name]: value });
  };

  const handleApplyFilters = async () => {
    try {
      const data = await getRFMMetrics(filters);
      setRfmData(data);
    } catch (err) {
      setError('Failed to apply filters');
    }
  };

  const handleExport = async () => {
    try {
      await exportCustomersCSV(filters);
      alert('CSV export started!');
    } catch (err) {
      setError('Failed to export CSV');
    }
  };

  if (loading) return <div className="dashboard">Loading...</div>;
  if (error) return <div className="dashboard error">{error}</div>;

  return (
    <div className="dashboard">
      <h1>Sales Dashboard</h1>
      
      {/* Filter Section */}
      <div className="filter-section">
        <h2>Filters</h2>
        <select name="segment" value={filters.segment} onChange={handleFilterChange}>
          <option value="">All Segments</option>
          <option value="VIP">VIP</option>
          <option value="High Value">High Value</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
        <button onClick={handleApplyFilters}>Apply Filters</button>
        <button onClick={handleExport}>Export to CSV</button>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        <div className="card">
          <h2>Customer Segments</h2>
          <pre>{JSON.stringify(segments, null, 2)}</pre>
        </div>

        <div className="card">
          <h2>RFM Metrics</h2>
          <pre>{JSON.stringify(rfmData, null, 2)}</pre>
        </div>

        <div className="card">
          <h2>Model Performance</h2>
          {modelMetrics && (
            <div>
              <p>Accuracy: {modelMetrics.accuracy}</p>
              <p>F1-Score: {modelMetrics.f1_score}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;