# Waste Management Admin Dashboard

A comprehensive admin dashboard for waste management systems built with React/Next.js frontend and Node.js backend. This application provides complete management of customers, collectors, waste collection data, and reward systems.

## 🚀 Features

### 📊 Dashboard Overview
- Real-time statistics and metrics
- Interactive charts and visualizations
- Performance analytics
- Recent activity feed
- Quick action buttons

### 👥 Customer Management
- Complete customer database
- Waste collection history
- Segregation accuracy tracking
- Complaint management
- Feedback and ratings
- Collection images and records

### 🚛 Collector Management
- Collector profiles and performance
- Attendance tracking
- Route management
- Complaint handling
- Performance metrics
- Assigned houses tracking

### 🎁 Reward System
- Customer reward management
- Points-based system
- High accuracy incentives
- Reward analytics
- Automated point calculation

### ⚙️ Admin Features
- System settings configuration
- User management
- Data export capabilities
- Advanced analytics
- Notification management

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Lucide React** - Icons
- **React Hot Toast** - Notifications

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **JWT** - Authentication
- **bcryptjs** - Password hashing
- **express-validator** - Input validation
- **cors** - Cross-origin resource sharing
- **helmet** - Security headers

## 📦 Installation

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Frontend Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   npm start
   ```

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start development server:**
   ```bash
   npm run dev
   ```

5. **Start production server:**
   ```bash
   npm start
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
PORT=5000
NODE_ENV=development
JWT_SECRET=your-super-secret-jwt-key
FRONTEND_URL=http://localhost:3000
```

### Default Login Credentials

- **Email:** admin@waste.com
- **Password:** password

## 📱 Usage

### Dashboard
- View real-time statistics
- Monitor collection progress
- Track performance metrics
- Access quick actions

### Customer Management
- Search and filter customers
- View detailed customer profiles
- Track waste collection history
- Manage complaints and feedback
- Monitor segregation accuracy

### Collector Management
- View collector performance
- Track attendance records
- Monitor assigned routes
- Handle complaints
- View collection statistics

### Reward System
- Create and manage rewards
- Award points to customers
- Track reward redemptions
- View reward analytics
- Configure reward settings

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/login` - Admin login
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout

### Customers
- `GET /api/customers` - Get all customers
- `GET /api/customers/:id` - Get customer details
- `PUT /api/customers/:id` - Update customer
- `GET /api/customers/:id/collection-history` - Get collection history
- `GET /api/customers/:id/complaints` - Get customer complaints

### Collectors
- `GET /api/collectors` - Get all collectors
- `GET /api/collectors/:id` - Get collector details
- `PUT /api/collectors/:id` - Update collector
- `GET /api/collectors/:id/attendance` - Get attendance history
- `GET /api/collectors/:id/complaints` - Get collector complaints

### Admin
- `GET /api/admin/rewards` - Get all rewards
- `POST /api/admin/rewards` - Create new reward
- `PUT /api/admin/rewards/:id` - Update reward
- `DELETE /api/admin/rewards/:id` - Delete reward
- `POST /api/admin/award-points/:customerId` - Award points
- `POST /api/admin/redeem-reward/:customerId` - Redeem reward

### Dashboard
- `GET /api/dashboard/overview` - Get dashboard overview
- `GET /api/dashboard/weekly-stats` - Get weekly statistics
- `GET /api/dashboard/waste-distribution` - Get waste distribution
- `GET /api/dashboard/recent-activity` - Get recent activity

## 🎨 Customization

### Styling
The application uses Tailwind CSS for styling. You can customize the theme by modifying:
- `tailwind.config.js` - Color scheme and theme
- `app/globals.css` - Global styles and components

### Components
All components are located in the `components/` directory and can be easily modified or extended.

### API Integration
The backend uses mock data for demonstration. To integrate with a real database:
1. Set up MongoDB or your preferred database
2. Update the route handlers to use database queries
3. Configure environment variables for database connection

## 🚀 Deployment

### Frontend Deployment (Vercel)
1. Push code to GitHub
2. Connect repository to Vercel
3. Configure environment variables
4. Deploy

### Backend Deployment (Heroku)
1. Create Heroku app
2. Set environment variables
3. Deploy using Git:
   ```bash
   heroku git:remote -a your-app-name
   git push heroku main
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team

## 🔮 Future Enhancements

- Real-time notifications
- Mobile app integration
- Advanced analytics dashboard
- Machine learning for waste prediction
- IoT device integration
- Multi-language support
- Advanced reporting features

---

**Built with ❤️ for sustainable waste management** 