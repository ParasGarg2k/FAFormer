import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Leaf, 
  Recycle, 
  Lightbulb, 
  TrendingUp, 
  Users, 
  Award,
  BookOpen,
  Play,
  ChevronRight
} from 'lucide-react';
import WasteClassifier from '../components/WasteClassifier';

const Home = () => {
  const [activeTab, setActiveTab] = useState('classifier');

  // Hardcoded blog data
  const blogs = [
    {
      id: 1,
      title: "The Impact of Proper Waste Segregation",
      excerpt: "Learn how proper waste segregation can reduce landfill waste by up to 60% and create a sustainable future.",
      image: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=250&fit=crop",
      category: "Education",
      readTime: "5 min read",
      date: "2024-01-15"
    },
    {
      id: 2,
      title: "Understanding Different Waste Categories",
      excerpt: "A comprehensive guide to wet, dry, red, and mixed waste categories and how to identify them correctly.",
      image: "https://images.unsplash.com/photo-1581578731548-c64695cc6952?w=400&h=250&fit=crop",
      category: "Guide",
      readTime: "8 min read",
      date: "2024-01-12"
    },
    {
      id: 3,
      title: "AI Technology in Waste Management",
      excerpt: "Discover how artificial intelligence is revolutionizing waste classification and recycling processes.",
      image: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=400&h=250&fit=crop",
      category: "Technology",
      readTime: "6 min read",
      date: "2024-01-10"
    }
  ];

  // Waste categories info
  const wasteCategories = [
    {
      name: "Wet Waste",
      color: "waste-wet",
      description: "Organic waste like food scraps, vegetable peels",
      icon: "🍎",
      examples: ["Food waste", "Vegetable peels", "Tea leaves", "Eggshells"]
    },
    {
      name: "Dry Waste",
      color: "waste-dry",
      description: "Recyclable materials like paper, plastic, metal",
      icon: "📦",
      examples: ["Paper", "Plastic bottles", "Metal cans", "Cardboard"]
    },
    {
      name: "Red Waste",
      color: "waste-red",
      description: "Hazardous waste requiring special handling",
      icon: "⚠️",
      examples: ["Batteries", "Medicines", "Thermometers", "Syringes"]
    },
    {
      name: "Mixed Waste",
      color: "waste-mixed",
      description: "Non-segregated waste that goes to landfill",
      icon: "🗑️",
      examples: ["Mixed garbage", "Contaminated items", "Non-recyclable"]
    }
  ];

  // Stats data
  const stats = [
    { label: "Waste Segregated", value: "2.5kg", icon: Recycle, color: "text-green-600" },
    { label: "Accuracy Rate", value: "94%", icon: Award, color: "text-blue-600" },
    { label: "Community Members", value: "1,247", icon: Users, color: "text-purple-600" },
    { label: "CO2 Saved", value: "12.3kg", icon: Leaf, color: "text-emerald-600" }
  ];

  const handleClassificationComplete = (result) => {
    console.log('Classification completed:', result);
    // Here you would typically save to backend
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Hero Section */}
      <motion.div 
        className="text-center py-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          Smart Waste Management for a
          <span className="gradient-text"> Sustainable Future</span>
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Use AI-powered waste classification to properly segregate your waste and contribute to a cleaner environment.
        </p>
        
        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              className="card p-4 text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <stat.icon className={`w-8 h-8 mx-auto mb-2 ${stat.color}`} />
              <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
              <div className="text-sm text-gray-600">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Main Content Tabs */}
      <div className="card">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { id: 'classifier', label: 'AI Classifier', icon: Play },
              { id: 'education', label: 'Waste Education', icon: BookOpen },
              { id: 'blogs', label: 'Latest Blogs', icon: TrendingUp }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* AI Classifier Tab */}
          {activeTab === 'classifier' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <WasteClassifier onClassificationComplete={handleClassificationComplete} />
            </motion.div>
          )}

          {/* Waste Education Tab */}
          {activeTab === 'education' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="space-y-8"
            >
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  Understanding Waste Categories
                </h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  Learn about different types of waste and how to properly segregate them for better recycling and disposal.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {wasteCategories.map((category, index) => (
                  <motion.div
                    key={category.name}
                    className={`waste-card ${category.color} p-6`}
                    initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <span className="text-2xl">{category.icon}</span>
                      <h3 className="text-xl font-bold">{category.name}</h3>
                    </div>
                    <p className="mb-4 opacity-90">{category.description}</p>
                    <div>
                      <h4 className="font-semibold mb-2">Examples:</h4>
                      <ul className="space-y-1 text-sm opacity-90">
                        {category.examples.map((example, idx) => (
                          <li key={idx}>• {example}</li>
                        ))}
                      </ul>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Tips Section */}
              <div className="card p-6 mt-8">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                  <Lightbulb className="w-6 h-6 text-yellow-500 mr-2" />
                  Pro Tips for Better Segregation
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2"></div>
                      <p className="text-gray-700">Rinse containers before disposal to avoid contamination</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2"></div>
                      <p className="text-gray-700">Keep separate bins for different waste types</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2"></div>
                      <p className="text-gray-700">Use the AI classifier when unsure about waste type</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2"></div>
                      <p className="text-gray-700">Compost organic waste when possible</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2"></div>
                      <p className="text-gray-700">Educate family members about proper segregation</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2"></div>
                      <p className="text-gray-700">Report hazardous waste to proper authorities</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Blogs Tab */}
          {activeTab === 'blogs' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  Latest from Our Blog
                </h2>
                <p className="text-gray-600">
                  Stay updated with the latest trends and tips in waste management
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {blogs.map((blog, index) => (
                  <motion.article
                    key={blog.id}
                    className="card overflow-hidden hover:shadow-xl transition-shadow duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    <img
                      src={blog.image}
                      alt={blog.title}
                      className="w-full h-48 object-cover"
                    />
                    <div className="p-6">
                      <div className="flex items-center space-x-2 mb-3">
                        <span className="px-2 py-1 bg-primary-100 text-primary-700 text-xs font-medium rounded-full">
                          {blog.category}
                        </span>
                        <span className="text-gray-500 text-sm">{blog.readTime}</span>
                      </div>
                      <h3 className="text-lg font-bold text-gray-900 mb-2 line-clamp-2">
                        {blog.title}
                      </h3>
                      <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                        {blog.excerpt}
                      </p>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-500 text-sm">
                          {new Date(blog.date).toLocaleDateString()}
                        </span>
                        <button className="flex items-center space-x-1 text-primary-600 hover:text-primary-700 font-medium">
                          <span>Read More</span>
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </motion.article>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home; 