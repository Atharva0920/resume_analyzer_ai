import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileText, Brain, Download, Eye, Zap, Target, Users, Award, TrendingUp, AlertCircle, CheckCircle, Star, BarChart3, PieChart, Activity } from 'lucide-react';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Real API integration with Python backend
const analyzeResume = async (text, industry = 'software_engineering') => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        industry: industry
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.success) {
      return data.analysis;
    } else {
      throw new Error(data.error || 'Analysis failed');
    }
  } catch (error) {
    console.error('Analysis error:', error);
    
    // Fallback to mock data in development if API is not available
    if (process.env.NODE_ENV === 'development') {
      console.warn('Using mock data - ensure backend is running on', API_BASE_URL);
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return {
        contact_info: {
          email: 'john.doe@email.com',
          phone: '+1-555-0123',
          linkedin: 'linkedin.com/in/johndoe',
          github: 'github.com/johndoe'
        },
        skills: {
          programming: ['python', 'javascript', 'react', 'sql'],
          frameworks: ['django', 'flask', 'tensorflow'],
          cloud: ['aws', 'docker', 'kubernetes'],
          ai_ml: ['machine learning', 'deep learning', 'nlp'],
          soft_skills: ['leadership', 'communication', 'problem solving']
        },
        experience_years: 5,
        education_level: 'bachelors',
        ats_score: 78.5,
        keyword_density: {
          'python': 2.3,
          'machine learning': 1.8,
          'react': 1.2,
          'leadership': 0.8,
          'sql': 1.5
        },
        sentiment_score: 0.82,
        readability_score: 74.2,
        industry_alignment: {
          software_engineering: 85.3,
          data_science: 72.1,
          product_management: 45.2,
          marketing: 32.1,
          finance: 28.7
        },
        recommendations: [
          "🔧 Improve ATS compatibility by adding clear section headers",
          "💡 Add more cloud computing skills to strengthen your profile",
          "🔍 Consider increasing mentions of: artificial intelligence, deployment",
          "📈 Highlight specific project outcomes and metrics",
          "✨ Use more action verbs and achievement-focused language"
        ],
        missing_sections: ['certifications', 'projects']
      };
    }
    
    throw error;
  }
};

// File upload function for real file processing
const analyzeFile = async (file, industry = 'software_engineering') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('industry', industry);

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.success) {
      return {
        analysis: data.analysis,
        extractedText: data.text_length ? `Successfully extracted ${data.text_length} characters` : ''
      };
    } else {
      throw new Error(data.error || 'File analysis failed');
    }
  } catch (error) {
    console.error('File analysis error:', error);
    throw error;
  }
};

// Health check function
const checkAPIHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (response.ok) {
      const data = await response.json();
      return data;
    }
    return null;
  } catch (error) {
    console.warn('API health check failed:', error);
    return null;
  }
};

const ResumeAnalyzerSPA = () => {
  const [file, setFile] = useState(null);
  const [resumeText, setResumeText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedIndustry, setSelectedIndustry] = useState('software_engineering');
  const [apiStatus, setApiStatus] = useState(null);
  const [isUsingMockData, setIsUsingMockData] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const fileInputRef = useRef(null);

  const industries = [
    { value: 'software_engineering', label: 'Software Engineering', icon: '💻' },
    { value: 'data_science', label: 'Data Science', icon: '📊' },
    { value: 'product_management', label: 'Product Management', icon: '🚀' },
    { value: 'marketing', label: 'Marketing', icon: '📢' },
    { value: 'finance', label: 'Finance', icon: '💰' }
  ];

  const handleFileUpload = useCallback(async (uploadedFile) => {
    setFile(uploadedFile);
    
    // If it's a real file, try to analyze it via API
    if (uploadedFile instanceof File) {
      try {
        setIsAnalyzing(true);
        const result = await analyzeFile(uploadedFile, selectedIndustry);
        setResumeText(result.extractedText || 'File processed successfully');
        setAnalysis(result.analysis);
        setActiveTab('results');
        return;
      } catch (error) {
        console.error('File upload failed:', error);
        alert(`Failed to process file: ${error.message}`);
        setIsAnalyzing(false);
        return;
      }
    }
    
    // Fallback to mock data for demo
    const mockResumeText = `John Doe
Software Engineer
Email: john.doe@email.com | Phone: +1-555-0123
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

SUMMARY
Experienced Software Engineer with 5+ years of expertise in Python, JavaScript, and machine learning. 
Proven track record of developing scalable web applications and implementing AI solutions.

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2021 - Present
• Led development of machine learning platform serving 1M+ users
• Implemented microservices architecture using Python, Django, and AWS
• Mentored junior developers and improved team productivity by 40%

Software Engineer | StartupXYZ | 2019 - 2021
• Built React-based frontend applications with modern JavaScript
• Developed RESTful APIs using Flask and PostgreSQL
• Collaborated with cross-functional teams using Agile methodologies

EDUCATION
Bachelor of Science in Computer Science | University of Technology | 2019

SKILLS
Programming: Python, JavaScript, SQL, HTML/CSS
Frameworks: React, Django, Flask, TensorFlow
Cloud: AWS, Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis`;

    setResumeText(mockResumeText);
    setActiveTab('analysis');
  }, [selectedIndustry]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileUpload(droppedFile);
    }
  }, [handleFileUpload]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const handleAnalyze = async () => {
    if (!resumeText.trim()) return;
    
    setIsAnalyzing(true);
    try {
      const result = await analyzeResume(resumeText, selectedIndustry);
      setAnalysis(result);
      setActiveTab('results');
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const ScoreCard = ({ title, score, icon: Icon, suffix = '' }) => (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Icon className="w-5 h-5 text-blue-600" />
          </div>
          <h3 className="font-semibold text-gray-800">{title}</h3>
        </div>
      </div>
      <div className={`text-3xl font-bold ${getScoreColor(score)} rounded-lg px-4 py-2 text-center`}>
        {score}{suffix}
      </div>
    </div>
  );

  const SkillBadge = ({ skill, category }) => {
    const colors = {
      programming: 'bg-blue-100 text-blue-800',
      frameworks: 'bg-green-100 text-green-800',
      cloud: 'bg-purple-100 text-purple-800',
      ai_ml: 'bg-red-100 text-red-800',
      soft_skills: 'bg-yellow-100 text-yellow-800'
    };
    
    return (
      <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${colors[category] || 'bg-gray-100 text-gray-800'} mr-2 mb-2`}>
        {skill}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">AI Resume Analyzer</h1>
                <p className="text-gray-600">Optimize your resume with AI-powered insights</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Star className="w-4 h-4 text-yellow-500" />
                <span>AI-Powered</span>
              </div>
              {/* API Status Indicator */}
              <div className="flex items-center space-x-2 text-xs">
                <div className={`w-2 h-2 rounded-full ${apiStatus ? 'bg-green-500' : 'bg-yellow-500'}`} />
                <span className="text-gray-500">
                  {apiStatus ? 'API Connected' : isUsingMockData ? 'Demo Mode' : 'Connecting...'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-xl">
          {[
            { id: 'upload', label: 'Upload Resume', icon: Upload },
            { id: 'analysis', label: 'Configure Analysis', icon: Target },
            { id: 'results', label: 'Results & Insights', icon: BarChart3 }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                activeTab === id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Upload Your Resume</h2>
              <p className="text-gray-600 mb-8">Support for PDF, DOCX, HTML, and TXT files</p>
            </div>

            {/* File Upload Area */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center hover:border-blue-400 hover:bg-blue-50 transition-all cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-6">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Drag and drop your resume here
              </h3>
              <p className="text-gray-600 mb-4">or click to browse files</p>
              <div className="flex justify-center space-x-4 text-sm text-gray-500">
                <span className="bg-gray-100 px-3 py-1 rounded-full">PDF</span>
                <span className="bg-gray-100 px-3 py-1 rounded-full">DOCX</span>
                <span className="bg-gray-100 px-3 py-1 rounded-full">HTML</span>
                <span className="bg-gray-100 px-3 py-1 rounded-full">TXT</span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.doc,.html,.htm,.txt"
                onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
                className="hidden"
              />
            </div>

            {/* Sample Resume Option */}
            <div className="text-center">
              <p className="text-gray-600 mb-4">Don't have a resume ready?</p>
              <button
                onClick={() => handleFileUpload({ name: 'sample_resume.txt' })}
                className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105"
              >
                <FileText className="w-4 h-4" />
                <span>Try Sample Resume</span>
              </button>
            </div>

            {file && (
              <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <div>
                    <h4 className="font-semibold text-green-900">File uploaded successfully!</h4>
                    <p className="text-green-700">Ready to analyze: {file.name}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analysis Configuration Tab */}
        {activeTab === 'analysis' && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Configure Your Analysis</h2>
              <p className="text-gray-600 mb-8">Customize the analysis based on your target industry</p>
            </div>

            {/* Industry Selection */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-6">Target Industry</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {industries.map((industry) => (
                  <button
                    key={industry.value}
                    onClick={() => setSelectedIndustry(industry.value)}
                    className={`p-6 rounded-xl border-2 transition-all hover:scale-105 ${
                      selectedIndustry === industry.value
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-3xl mb-3">{industry.icon}</div>
                    <div className="font-medium text-sm">{industry.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Resume Preview */}
            {resumeText && (
              <div className="bg-white rounded-2xl shadow-lg p-8">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-semibold text-gray-900">Resume Preview</h3>
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Eye className="w-4 h-4" />
                    <span>{resumeText.length} characters</span>
                  </div>
                </div>
                <div className="bg-gray-50 rounded-xl p-6 max-h-64 overflow-y-auto">
                  <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono">
                    {resumeText.substring(0, 500)}
                    {resumeText.length > 500 && '...'}
                  </pre>
                </div>
              </div>
            )}

            {/* Analyze Button */}
            <div className="text-center">
              <button
                onClick={handleAnalyze}
                disabled={!resumeText.trim() || isAnalyzing}
                className="inline-flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {isAnalyzing ? (
                  <>
                    <Activity className="w-5 h-5 animate-spin" />
                    <span>Analyzing Resume...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    <span>Analyze with AI</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && analysis && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Analysis Results</h2>
              <p className="text-gray-600 mb-8">AI-powered insights and recommendations for your resume</p>
            </div>

            {/* Score Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <ScoreCard
                title="ATS Score"
                score={analysis.ats_score}
                icon={Target}
                suffix="/100"
              />
              <ScoreCard
                title="Readability"
                score={analysis.readability_score}
                icon={Eye}
                suffix="/100"
              />
              <ScoreCard
                title="Sentiment"
                score={Math.round(analysis.sentiment_score * 100)}
                icon={TrendingUp}
                suffix="/100"
              />
              <ScoreCard
                title="Experience"
                score={analysis.experience_years}
                icon={Award}
                suffix=" years"
              />
            </div>

            {/* Skills Analysis */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Users className="w-5 h-5 mr-2 text-blue-600" />
                Skills Analysis
              </h3>
              <div className="space-y-6">
                {Object.entries(analysis.skills).map(([category, skills]) => (
                  <div key={category}>
                    <h4 className="font-medium text-gray-800 mb-3 capitalize">
                      {category.replace('_', ' ')} ({skills.length})
                    </h4>
                    <div className="flex flex-wrap">
                      {skills.map((skill, index) => (
                        <SkillBadge key={index} skill={skill} category={category} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Industry Alignment */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <PieChart className="w-5 h-5 mr-2 text-blue-600" />
                Industry Alignment
              </h3>
              <div className="space-y-4">
                {Object.entries(analysis.industry_alignment)
                  .sort(([,a], [,b]) => b - a)
                  .map(([industry, score]) => (
                    <div key={industry} className="flex items-center justify-between">
                      <span className="capitalize font-medium text-gray-700">
                        {industry.replace('_', ' ')}
                      </span>
                      <div className="flex items-center space-x-3">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                            style={{ width: `${score}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-gray-900 w-12">
                          {score}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <AlertCircle className="w-5 h-5 mr-2 text-yellow-600" />
                AI Recommendations
              </h3>
              <div className="space-y-4">
                {analysis.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start space-x-3 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                    <div className="flex-shrink-0 w-6 h-6 bg-yellow-100 rounded-full flex items-center justify-center text-yellow-600 font-semibold text-sm">
                      {index + 1}
                    </div>
                    <p className="text-gray-800">{rec}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Contact Information */}
            {analysis.contact_info && Object.keys(analysis.contact_info).length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg p-8">
                <h3 className="text-xl font-semibold text-gray-900 mb-6">Contact Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(analysis.contact_info).map(([key, value]) => (
                    <div key={key} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                      <span className="font-medium text-gray-600 capitalize">{key}:</span>
                      <span className="text-gray-900">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Download Report */}
            <div className="text-center">
              <button className="inline-flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors">
                <Download className="w-4 h-4" />
                <span>Download Full Report</span>
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Brain className="w-8 h-8 text-blue-600 animate-pulse" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">AI Analysis in Progress</h3>
                <p className="text-gray-600 mb-6">Our AI is analyzing your resume...</p>
                <div className="flex justify-center">
                  <div className="flex space-x-1">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
                        style={{ animationDelay: `${i * 0.1}s` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResumeAnalyzerSPA;