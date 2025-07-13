const mongoose = require('mongoose');
const User = require('./models/User');
const Reward = require('./models/Reward');
const dotenv = require('dotenv');
dotenv.config();

const MONGODB_URI = process.env.MONGODB_URI;

async function seed() {
  await mongoose.connect(MONGODB_URI);

  // Clear existing data
  await User.deleteMany({});
  await Reward.deleteMany({});

  // Create admin
  await User.create({
    name: 'Admin User',
    email: 'admin@waste.com',
    password: '$2a$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', // 'password'
    role: 'admin',
    status: 'active',
    rewards: 0,
    accuracy: 0,
    totalWaste: 0,
    feedback: 0
  });

  // Create Indian customers
  await User.insertMany([
    {
      name: 'Amit Sharma',
      email: 'amit.sharma@email.com',
      phone: '+91 98765-43210',
      role: 'customer',
      address: '12 MG Road, Delhi',
      status: 'active',
      rewards: 80,
      accuracy: 78,
      totalWaste: 38.2,
      feedback: 4.1,
      lastCollection: new Date('2024-07-10')
    },
    {
      name: 'Priya Singh',
      email: 'priya.singh@email.com',
      phone: '+91 98765-43211',
      role: 'customer',
      address: '45 Park Street, Kolkata',
      status: 'active',
      rewards: 120,
      accuracy: 82,
      totalWaste: 42.7,
      feedback: 4.3,
      lastCollection: new Date('2024-07-09')
    },
    {
      name: 'Rahul Verma',
      email: 'rahul.verma@email.com',
      phone: '+91 98765-43212',
      role: 'customer',
      address: '78 Residency Road, Bengaluru',
      status: 'active',
      rewards: 60,
      accuracy: 75,
      totalWaste: 29.5,
      feedback: 3.9,
      lastCollection: new Date('2024-07-08')
    },
    {
      name: 'Sneha Iyer',
      email: 'sneha.iyer@email.com',
      phone: '+91 98765-43213',
      role: 'customer',
      address: '23 Anna Salai, Chennai',
      status: 'active',
      rewards: 95,
      accuracy: 80,
      totalWaste: 36.1,
      feedback: 4.0,
      lastCollection: new Date('2024-07-07')
    },
    {
      name: 'Vikas Patel',
      email: 'vikas.patel@email.com',
      phone: '+91 98765-43214',
      role: 'customer',
      address: '56 CG Road, Ahmedabad',
      status: 'active',
      rewards: 110,
      accuracy: 73,
      totalWaste: 41.3,
      feedback: 3.7,
      lastCollection: new Date('2024-07-06')
    }
  ]);

  // Create Indian collectors
  await User.insertMany([
    {
      name: 'Ramesh Kumar',
      email: 'ramesh.kumar@waste.com',
      phone: '+91 90000-11111',
      role: 'collector',
      address: 'Sector 21, Noida',
      status: 'active',
      rewards: 0,
      accuracy: 81,
      totalWaste: 1200.5,
      feedback: 4.0,
      lastCollection: new Date('2024-07-10')
    },
    {
      name: 'Sunita Joshi',
      email: 'sunita.joshi@waste.com',
      phone: '+91 90000-11112',
      role: 'collector',
      address: 'Baner, Pune',
      status: 'active',
      rewards: 0,
      accuracy: 77,
      totalWaste: 980.3,
      feedback: 4.2,
      lastCollection: new Date('2024-07-09')
    },
    {
      name: 'Deepak Mehta',
      email: 'deepak.mehta@waste.com',
      phone: '+91 90000-11113',
      role: 'collector',
      address: 'Banjara Hills, Hyderabad',
      status: 'active',
      rewards: 0,
      accuracy: 74,
      totalWaste: 1100.7,
      feedback: 3.8,
      lastCollection: new Date('2024-07-08')
    }
  ]);

  // Create rewards
  await Reward.insertMany([
    {
      id: 'R001',
      name: '10% Discount on Next Bill',
      description: 'Get 10% discount on your next waste collection bill',
      pointsRequired: 100,
      discount: 10,
      isActive: true,
      customersEligible: 45,
      createdAt: new Date('2024-01-01'),
      totalRedeemed: 23
    },
    {
      id: 'R002',
      name: 'Free Collection Week',
      description: 'One week of free waste collection service',
      pointsRequired: 200,
      discount: 100,
      isActive: true,
      customersEligible: 23,
      createdAt: new Date('2024-01-01'),
      totalRedeemed: 12
    },
    {
      id: 'R003',
      name: 'Premium Bin Upgrade',
      description: 'Upgrade to a premium waste collection bin',
      pointsRequired: 300,
      discount: 50,
      isActive: true,
      customersEligible: 12,
      createdAt: new Date('2024-01-01'),
      totalRedeemed: 8
    },
    {
      id: 'R004',
      name: 'Monthly Gift Card',
      description: 'Receive a ₹2000 gift card for local stores',
      pointsRequired: 500,
      discount: 25,
      isActive: false,
      customersEligible: 8,
      createdAt: new Date('2024-01-01'),
      totalRedeemed: 5
    }
  ]);

  console.log('Dummy data seeded!');
  await mongoose.disconnect();
}

seed(); 