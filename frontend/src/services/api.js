// API Service - Contains all fetch functions for backend endpoints
const API_BASE_URL = 'http://localhost:5000/api';

// Helper function for error handling
const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'API request failed');
  }
  return response.json();
};

// 1. Get all customer segments and their distribution
export const getCustomerSegments = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/customers/segments`);
    return await handleResponse(response);
  } catch (error) {
    console.error('Error fetching segments:', error);
    throw error;
  }
};

// 2. Get RFM metrics with optional filters
export const getRFMMetrics = async (filters = {}) => {
  try {
    const queryParams = new URLSearchParams();
    
    if (filters.segment) queryParams.append('segment', filters.segment);
    if (filters.minRecency) queryParams.append('min_recency', filters.minRecency);
    if (filters.maxRecency) queryParams.append('max_recency', filters.maxRecency);
    if (filters.minFrequency) queryParams.append('min_frequency', filters.minFrequency);
    if (filters.maxFrequency) queryParams.append('max_frequency', filters.maxFrequency);
    if (filters.minMonetary) queryParams.append('min_monetary', filters.minMonetary);
    if (filters.maxMonetary) queryParams.append('max_monetary', filters.maxMonetary);
    
    const url = `${API_BASE_URL}/customers/rfm${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    const response = await fetch(url);
    return await handleResponse(response);
  } catch (error) {
    console.error('Error fetching RFM metrics:', error);
    throw error;
  }
};

// 3. Get individual customer details
export const getCustomerDetails = async (customerId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/customers/${customerId}/details`);
    return await handleResponse(response);
  } catch (error) {
    console.error(`Error fetching customer ${customerId} details:`, error);
    throw error;
  }
};

// 4. Get model performance metrics
export const getModelMetrics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/model/metrics`);
    return await handleResponse(response);
  } catch (error) {
    console.error('Error fetching model metrics:', error);
    throw error;
  }
};

// 5. Export customers to CSV
export const exportCustomersCSV = async (filters = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/export/customers`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(filters),
    });
    
    if (!response.ok) {
      throw new Error('Export failed');
    }
    
    // Get the CSV file as a blob
    const blob = await response.blob();
    
    // Create a download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a'); 
    link.href = url;
    link.setAttribute('download', `customers_export_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    
    // Cleanup
    window.URL.revokeObjectURL(url);
    link.remove();
  } catch (error) {
    console.error('Error exporting customers:', error);
    throw error;
  }
};
