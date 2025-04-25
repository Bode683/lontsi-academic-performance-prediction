document.addEventListener('DOMContentLoaded', function() {
    // Tab functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === tabId) {
                    content.classList.add('active');
                }
            });
        });
    });
    
    // Form submission handling
    const predictionForm = document.getElementById('prediction-form');
    
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        document.getElementById('loading-overlay').classList.remove('hidden');
        
        // Hide previous results and errors
        document.getElementById('error-message').classList.add('hidden');
        document.getElementById('result-section').classList.add('hidden');
        
        // Collect form data
        const formData = new FormData(predictionForm);
        const formObject = {};
        
        formData.forEach((value, key) => {
            formObject[key] = value;
        });
        
        // Send data to API
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formObject)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            displaySingleResult(data);
        })
        .catch(error => {
            displayError(error.message);
        })
        .finally(() => {
            document.getElementById('loading-overlay').classList.add('hidden');
        });
    });
    
    // JSON file upload handling
    const jsonFileInput = document.getElementById('json-file-input');
    const jsonTextarea = document.getElementById('json-input');
    
    jsonFileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                jsonTextarea.value = e.target.result;
                
                // Check if valid JSON and format it
                try {
                    const json = JSON.parse(e.target.result);
                    jsonTextarea.value = JSON.stringify(json, null, 2);
                } catch (error) {
                    // Keep as is if not valid JSON
                }
            };
            
            reader.readAsText(file);
        }
    });
    
    // JSON submission handling
    const jsonSubmitBtn = document.getElementById('json-submit-btn');
    
    jsonSubmitBtn.addEventListener('click', function() {
        let jsonData;
        
        try {
            jsonData = JSON.parse(jsonTextarea.value);
        } catch (error) {
            displayError('Invalid JSON format: ' + error.message);
            return;
        }
        
        // Show loading indicator
        document.getElementById('loading-overlay').classList.remove('hidden');
        
        // Hide previous results and errors
        document.getElementById('error-message').classList.add('hidden');
        document.getElementById('result-section').classList.add('hidden');
        
        // Send data to API
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jsonData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // Check if it's a batch result (array) or single result
            if (Array.isArray(data)) {
                displayBatchResults(data);
            } else {
                displaySingleResult(data);
            }
        })
        .catch(error => {
            displayError(error.message);
        })
        .finally(() => {
            document.getElementById('loading-overlay').classList.add('hidden');
        });
    });
    
    // Function to display a single prediction result
    function displaySingleResult(result) {
        if (result.error) {
            displayError(result.error);
            return;
        }
        
        // Show result section
        document.getElementById('result-section').classList.remove('hidden');
        document.getElementById('single-result').classList.remove('hidden');
        document.getElementById('batch-results').classList.add('hidden');
        
        // Update grade display
        document.getElementById('grade-value').textContent = result.predicted_grade;
        
        // Set grade color based on the grade
        const gradeElement = document.getElementById('grade-value');
        const grade = result.predicted_grade;
        
        switch(grade) {
            case 'A':
                gradeElement.style.color = '#2ecc71'; // Green
                break;
            case 'B':
                gradeElement.style.color = '#3498db'; // Blue
                break;
            case 'C':
                gradeElement.style.color = '#f39c12'; // Orange
                break;
            case 'D':
                gradeElement.style.color = '#e67e22'; // Dark Orange
                break;
            case 'F':
                gradeElement.style.color = '#e74c3c'; // Red
                break;
            default:
                gradeElement.style.color = '#2c3e50'; // Default dark
        }
        
        // Generate confidence bars
        const confidenceBars = document.getElementById('confidence-bars');
        confidenceBars.innerHTML = '';
        
        const grades = ['A', 'B', 'C', 'D', 'F'];
        grades.forEach(grade => {
            const confidence = result.confidence[grade] || 0;
            const confidencePercentage = (confidence * 100).toFixed(0) + '%';
            
            const barDiv = document.createElement('div');
            barDiv.className = 'confidence-bar';
            barDiv.innerHTML = `
                <div class="bar-label">${grade}</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${confidencePercentage}"></div>
                </div>
                <div class="bar-value">${confidencePercentage}</div>
            `;
            
            confidenceBars.appendChild(barDiv);
        });
        
        // Scroll to results
        document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to display batch prediction results
    function displayBatchResults(results) {
        if (!results.length) {
            displayError('No results returned');
            return;
        }
        
        // Show batch results section
        document.getElementById('result-section').classList.remove('hidden');
        document.getElementById('single-result').classList.add('hidden');
        document.getElementById('batch-results').classList.remove('hidden');
        
        // Update batch count
        document.getElementById('batch-count').textContent = results.length;
        
        // Calculate grade distribution
        const gradeCount = { 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0 };
        results.forEach(result => {
            const grade = result.predicted_grade;
            if (gradeCount[grade] !== undefined) {
                gradeCount[grade]++;
            }
        });
        
        // Display grade distribution
        const distributionText = Object.entries(gradeCount)
            .filter(([_, count]) => count > 0)
            .map(([grade, count]) => `${grade}: ${count}`)
            .join(', ');
        
        document.getElementById('grade-distribution').textContent = distributionText;
        
        // Populate batch table
        const tableBody = document.getElementById('batch-results-body');
        tableBody.innerHTML = '';
        
        results.forEach((result, index) => {
            const row = document.createElement('tr');
            
            // Find the highest confidence value
            const highestConfidence = Math.max(...Object.values(result.confidence));
            const confidencePercentage = (highestConfidence * 100).toFixed(0) + '%';
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.predicted_grade}</td>
                <td>${confidencePercentage}</td>
                <td>
                    <button class="view-details-btn" data-index="${index}">View Details</button>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Add event listeners for "View Details" buttons
        document.querySelectorAll('.view-details-btn').forEach(button => {
            button.addEventListener('click', function() {
                const index = parseInt(this.getAttribute('data-index'));
                displaySingleResult(results[index]);
            });
        });
        
        // Scroll to results
        document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to display errors
    function displayError(message) {
        const errorElement = document.getElementById('error-message');
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
        errorElement.scrollIntoView({ behavior: 'smooth' });
    }
});
