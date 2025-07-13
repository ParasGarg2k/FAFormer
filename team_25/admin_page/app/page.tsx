'use client'

import { useState } from 'react'
import Dashboard from '@/components/Dashboard'
import CustomersTab from '@/components/CustomersTab'
import CollectorsTab from '@/components/CollectorsTab'
import AdminTab from '@/components/AdminTab'
import Sidebar from '@/components/Sidebar'
import Header from '@/components/Header'

export default function Home() {
  const [activeTab, setActiveTab] = useState('dashboard')

  const tabs = [
    { id: 'dashboard', name: 'Dashboard', icon: '📊' },
    { id: 'customers', name: 'Customers', icon: '👥' },
    { id: 'collectors', name: 'Collectors', icon: '🚛' },
    { id: 'admin', name: 'Admin', icon: '⚙️' },
  ]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />
      case 'customers':
        return <CustomersTab />
      case 'collectors':
        return <CollectorsTab />
      case 'admin':
        return <AdminTab />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} tabs={tabs} />
        <div className="flex-1">
          <Header />
          <main className="p-6">
            {renderTabContent()}
          </main>
        </div>
      </div>
    </div>
  )
} 