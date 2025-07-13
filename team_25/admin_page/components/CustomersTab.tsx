'use client'

import { useState, useEffect } from 'react'
import { 
  Search, 
  Filter, 
  Eye, 
  Download, 
  Star,
  AlertTriangle,
  CheckCircle,
  Clock,
  Calendar,
  MapPin,
  X
} from 'lucide-react'

interface Customer {
  id: string
  name: string
  address: string
  phone: string
  email: string
  totalWaste: number
  accuracy: number
  complaints: number
  feedback: number
  lastCollection: string
  status: 'active' | 'inactive' | 'suspended'
  rewards: number
}

const hardcodedCustomers = [
  {
    id: 'C001',
    name: 'Amit Sharma',
    address: '12 MG Road, Delhi',
    phone: '+91 98765-43210',
    email: 'amit.sharma@email.com',
    totalWaste: 38.2,
    accuracy: 78,
    complaints: 1,
    feedback: 4.1,
    lastCollection: '2024-07-10',
    status: 'active' as const,
    rewards: 80
  },
  {
    id: 'C002',
    name: 'Priya Singh',
    address: '45 Park Street, Kolkata',
    phone: '+91 98765-43211',
    email: 'priya.singh@email.com',
    totalWaste: 42.7,
    accuracy: 82,
    complaints: 2,
    feedback: 4.3,
    lastCollection: '2024-07-09',
    status: 'active' as const,
    rewards: 120
  },
  {
    id: 'C003',
    name: 'Rahul Verma',
    address: '78 Residency Road, Bengaluru',
    phone: '+91 98765-43212',
    email: 'rahul.verma@email.com',
    totalWaste: 29.5,
    accuracy: 75,
    complaints: 0,
    feedback: 3.9,
    lastCollection: '2024-07-08',
    status: 'active' as const,
    rewards: 60
  },
  {
    id: 'C004',
    name: 'Sneha Iyer',
    address: '23 Anna Salai, Chennai',
    phone: '+91 98765-43213',
    email: 'sneha.iyer@email.com',
    totalWaste: 36.1,
    accuracy: 80,
    complaints: 1,
    feedback: 4.0,
    lastCollection: '2024-07-07',
    status: 'active' as const,
    rewards: 95
  },
  {
    id: 'C005',
    name: 'Vikas Patel',
    address: '56 CG Road, Ahmedabad',
    phone: '+91 98765-43214',
    email: 'vikas.patel@email.com',
    totalWaste: 41.3,
    accuracy: 73,
    complaints: 3,
    feedback: 3.7,
    lastCollection: '2024-07-06',
    status: 'active' as const,
    rewards: 110
  }
];

export default function CustomersTab() {
  const [customers, setCustomers] = useState<Customer[]>(hardcodedCustomers);
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const filteredCustomers = customers.filter(customer =>
    customer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    customer.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    customer.address.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-success-100 text-success-800'
      case 'inactive':
        return 'bg-gray-100 text-gray-800'
      case 'suspended':
        return 'bg-danger-100 text-danger-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 90) return 'text-success-600'
    if (accuracy >= 80) return 'text-warning-600'
    return 'text-danger-600'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading customers...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-red-600">Error: {error}</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Customers Management</h1>
          <p className="text-gray-600">Manage customer data, waste collection, and feedback</p>
        </div>
        <button className="btn-primary">
          <Download className="w-4 h-4 mr-2" />
          Export Data
        </button>
      </div>

      {/* Search and Filters */}
      <div className="card">
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search customers by name, email, or address..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent w-full"
            />
          </div>
          <button className="btn-secondary">
            <Filter className="w-4 h-4 mr-2" />
            Filters
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Customers</p>
              <p className="text-2xl font-bold text-gray-900">{customers.length}</p>
            </div>
            <div className="p-3 bg-primary-100 rounded-lg">
              <CheckCircle className="w-6 h-6 text-primary-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg. Accuracy</p>
              <p className="text-2xl font-bold text-gray-900">
                {customers.length > 0 ? Math.round(customers.reduce((acc, c) => acc + c.accuracy, 0) / customers.length) : 0}%
              </p>
            </div>
            <div className="p-3 bg-success-100 rounded-lg">
              <Star className="w-6 h-6 text-success-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Complaints</p>
              <p className="text-2xl font-bold text-gray-900">
                {customers.reduce((acc, c) => acc + c.complaints, 0)}
              </p>
            </div>
            <div className="p-3 bg-warning-100 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-warning-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Waste (kg)</p>
              <p className="text-2xl font-bold text-gray-900">
                {customers.reduce((acc, c) => acc + c.totalWaste, 0).toFixed(1)}
              </p>
            </div>
            <div className="p-3 bg-warning-100 rounded-lg">
              <Clock className="w-6 h-6 text-warning-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Customers Table */}
      <div className="card">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Customer</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Contact</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Waste (kg)</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Accuracy</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Status</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredCustomers.map((customer) => (
                <tr key={customer.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-4 px-4">
                    <div>
                      <p className="font-medium text-gray-900">{customer.name}</p>
                      <p className="text-sm text-gray-500 flex items-center">
                        <MapPin className="w-3 h-3 mr-1" />
                        {customer.address}
                      </p>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <div>
                      <p className="text-sm text-gray-900">{customer.email}</p>
                      <p className="text-sm text-gray-500">{customer.phone}</p>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <p className="font-medium text-gray-900">{customer.totalWaste.toFixed(1)} kg</p>
                  </td>
                  <td className="py-4 px-4">
                    <p className={`font-medium ${getAccuracyColor(customer.accuracy)}`}>
                      {customer.accuracy}%
                    </p>
                  </td>
                  <td className="py-4 px-4">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(customer.status)}`}>
                      {customer.status}
                    </span>
                  </td>
                  <td className="py-4 px-4">
                    <button
                      onClick={() => {
                        setSelectedCustomer(customer)
                        setShowDetails(true)
                      }}
                      className="text-primary-600 hover:text-primary-800"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Customer Details Modal */}
      {showDetails && selectedCustomer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-900">Customer Details</h2>
              <button
                onClick={() => setShowDetails(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Personal Information</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">Name:</span> {selectedCustomer.name}</p>
                  <p><span className="font-medium">Email:</span> {selectedCustomer.email}</p>
                  <p><span className="font-medium">Phone:</span> {selectedCustomer.phone}</p>
                  <p><span className="font-medium">Address:</span> {selectedCustomer.address}</p>
                </div>
              </div>
              
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Collection Data</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">Total Waste:</span> {selectedCustomer.totalWaste.toFixed(1)} kg</p>
                  <p><span className="font-medium">Accuracy:</span> {selectedCustomer.accuracy}%</p>
                  <p><span className="font-medium">Complaints:</span> {selectedCustomer.complaints}</p>
                  <p><span className="font-medium">Feedback Rating:</span> {selectedCustomer.feedback}/5</p>
                  <p><span className="font-medium">Rewards Points:</span> {selectedCustomer.rewards}</p>
                  <p><span className="font-medium">Last Collection:</span> {selectedCustomer.lastCollection}</p>
                </div>
              </div>
            </div>
            
            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => setShowDetails(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
              >
                Close
              </button>
              <button className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600">
                Edit Customer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 