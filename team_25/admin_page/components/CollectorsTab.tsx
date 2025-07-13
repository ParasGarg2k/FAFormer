'use client'

import { useState, useEffect } from 'react'
import { 
  Search, 
  Filter, 
  Eye, 
  Download, 
  Truck,
  AlertTriangle,
  CheckCircle,
  Clock,
  Calendar,
  MapPin,
  X,
  User,
  Phone,
  Mail,
  Star
} from 'lucide-react'

interface Collector {
  id: string
  name: string
  phone: string
  email: string
  assignedArea: string
  totalWasteCollected: number
  complaintsAgainst: number
  attendance: number
  assignedHouses: number
  lastActive: string
  status: 'active' | 'inactive' | 'on_leave'
  rating: number
  vehicleNumber: string
}

const hardcodedCollectors = [
  {
    id: 'COL001',
    name: 'Ramesh Kumar',
    phone: '+91 90000-11111',
    email: 'ramesh.kumar@waste.com',
    assignedArea: 'Sector 21, Noida',
    totalWasteCollected: 1200.5,
    complaintsAgainst: 2,
    attendance: 81,
    assignedHouses: 45,
    lastActive: '2024-07-10',
    status: 'active' as const,
    rating: 4.0,
    vehicleNumber: 'DL-01-AB-1234'
  },
  {
    id: 'COL002',
    name: 'Sunita Joshi',
    phone: '+91 90000-11112',
    email: 'sunita.joshi@waste.com',
    assignedArea: 'Baner, Pune',
    totalWasteCollected: 980.3,
    complaintsAgainst: 1,
    attendance: 77,
    assignedHouses: 38,
    lastActive: '2024-07-09',
    status: 'active' as const,
    rating: 4.2,
    vehicleNumber: 'MH-12-XY-5678'
  },
  {
    id: 'COL003',
    name: 'Deepak Mehta',
    phone: '+91 90000-11113',
    email: 'deepak.mehta@waste.com',
    assignedArea: 'Banjara Hills, Hyderabad',
    totalWasteCollected: 1100.7,
    complaintsAgainst: 0,
    attendance: 74,
    assignedHouses: 42,
    lastActive: '2024-07-08',
    status: 'active' as const,
    rating: 3.8,
    vehicleNumber: 'TS-09-ZZ-4321'
  },
  {
    id: 'COL004',
    name: 'Lakshmi Devi',
    phone: '+91 90000-11114',
    email: 'lakshmi.devi@waste.com',
    assignedArea: 'Anna Nagar, Chennai',
    totalWasteCollected: 890.2,
    complaintsAgainst: 1,
    attendance: 79,
    assignedHouses: 35,
    lastActive: '2024-07-07',
    status: 'active' as const,
    rating: 4.1,
    vehicleNumber: 'TN-01-CD-9876'
  },
  {
    id: 'COL005',
    name: 'Rajesh Patel',
    phone: '+91 90000-11115',
    email: 'rajesh.patel@waste.com',
    assignedArea: 'Satellite, Ahmedabad',
    totalWasteCollected: 1050.8,
    complaintsAgainst: 3,
    attendance: 72,
    assignedHouses: 40,
    lastActive: '2024-07-06',
    status: 'active' as const,
    rating: 3.7,
    vehicleNumber: 'GJ-01-EF-5432'
  }
];

export default function CollectorsTab() {
  const [collectors, setCollectors] = useState<Collector[]>(hardcodedCollectors);
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCollector, setSelectedCollector] = useState<Collector | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const filteredCollectors = collectors.filter(collector =>
    collector.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    collector.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    collector.assignedArea.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-success-100 text-success-800'
      case 'inactive':
        return 'bg-gray-100 text-gray-800'
      case 'on_leave':
        return 'bg-warning-100 text-warning-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getRatingColor = (rating: number) => {
    if (rating >= 4.5) return 'text-success-600'
    if (rating >= 4.0) return 'text-warning-600'
    return 'text-danger-600'
  }

  const getAttendanceColor = (attendance: number) => {
    if (attendance >= 95) return 'text-success-600'
    if (attendance >= 90) return 'text-warning-600'
    return 'text-danger-600'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading collectors...</div>
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
          <h1 className="text-3xl font-bold text-gray-900">Collectors Management</h1>
          <p className="text-gray-600">Monitor collector performance, attendance, and complaints</p>
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
              placeholder="Search collectors by name, email, or area..."
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
              <p className="text-sm font-medium text-gray-600">Total Collectors</p>
              <p className="text-2xl font-bold text-gray-900">{collectors.length}</p>
            </div>
            <div className="p-3 bg-primary-100 rounded-lg">
              <Truck className="w-6 h-6 text-primary-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg. Rating</p>
              <p className="text-2xl font-bold text-gray-900">
                {collectors.length > 0 ? (collectors.reduce((acc, c) => acc + c.rating, 0) / collectors.length).toFixed(1) : '0.0'}
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
                {collectors.reduce((acc, c) => acc + c.complaintsAgainst, 0)}
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
                {collectors.reduce((acc, c) => acc + c.totalWasteCollected, 0).toFixed(1)}
              </p>
            </div>
            <div className="p-3 bg-warning-100 rounded-lg">
              <Clock className="w-6 h-6 text-warning-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Collectors Table */}
      <div className="card">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Collector</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Contact</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Area</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Waste (kg)</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Rating</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Status</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredCollectors.map((collector) => (
                <tr key={collector.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-4 px-4">
                    <div>
                      <p className="font-medium text-gray-900">{collector.name}</p>
                      <p className="text-sm text-gray-500">ID: {collector.id}</p>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <div>
                      <p className="text-sm text-gray-900">{collector.email}</p>
                      <p className="text-sm text-gray-500">{collector.phone}</p>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <p className="text-sm text-gray-900">{collector.assignedArea}</p>
                  </td>
                  <td className="py-4 px-4">
                    <p className="font-medium text-gray-900">{collector.totalWasteCollected.toFixed(1)} kg</p>
                  </td>
                  <td className="py-4 px-4">
                    <p className={`font-medium ${getRatingColor(collector.rating)}`}>
                      {collector.rating}/5
                    </p>
                  </td>
                  <td className="py-4 px-4">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(collector.status)}`}>
                      {collector.status}
                    </span>
                  </td>
                  <td className="py-4 px-4">
                    <button
                      onClick={() => {
                        setSelectedCollector(collector)
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

      {/* Collector Details Modal */}
      {showDetails && selectedCollector && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-900">Collector Details</h2>
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
                  <p><span className="font-medium">Name:</span> {selectedCollector.name}</p>
                  <p><span className="font-medium">ID:</span> {selectedCollector.id}</p>
                  <p><span className="font-medium">Email:</span> {selectedCollector.email}</p>
                  <p><span className="font-medium">Phone:</span> {selectedCollector.phone}</p>
                  <p><span className="font-medium">Vehicle Number:</span> {selectedCollector.vehicleNumber}</p>
                </div>
              </div>
              
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Performance Data</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">Assigned Area:</span> {selectedCollector.assignedArea}</p>
                  <p><span className="font-medium">Total Waste Collected:</span> {selectedCollector.totalWasteCollected.toFixed(1)} kg</p>
                  <p><span className="font-medium">Rating:</span> {selectedCollector.rating}/5</p>
                  <p><span className="font-medium">Attendance:</span> {selectedCollector.attendance}%</p>
                  <p><span className="font-medium">Assigned Houses:</span> {selectedCollector.assignedHouses}</p>
                  <p><span className="font-medium">Complaints Against:</span> {selectedCollector.complaintsAgainst}</p>
                  <p><span className="font-medium">Last Active:</span> {selectedCollector.lastActive}</p>
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
                Edit Collector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 