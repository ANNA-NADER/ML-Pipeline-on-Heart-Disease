# Task 2.9: Deployment using Ngrok

import subprocess
import sys
import os
import time
import threading
import requests
from datetime import datetime
import json

class NgrokDeployer:
    """Class to handle Ngrok deployment for Streamlit applications"""
    
    def __init__(self):
        self.ngrok_process = None
        self.streamlit_process = None
        self.public_url = None
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        # Check for Streamlit
        try:
            import streamlit
            print(f"✅ Streamlit {streamlit.__version__} found")
        except ImportError:
            print("❌ Streamlit not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed")
        
        # Check for required packages
        required_packages = ['plotly', 'pandas', 'numpy', 'joblib', 'scikit-learn']
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} found")
            except ImportError:
                print(f"❌ {package} not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed")
        
        print("📦 All dependencies are ready!")
        return True
    
    def install_ngrok(self):
        """Install ngrok if not already installed"""
        print("🔧 Setting up ngrok...")
        
        try:
            # Check if ngrok is already installed
            result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ngrok already installed")
                print(f"Version: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        # Install pyngrok as Python wrapper
        try:
            import pyngrok
            print("✅ pyngrok already installed")
        except ImportError:
            print("📦 Installing pyngrok...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
            print("✅ pyngrok installed")
        
        return True
    
    def setup_ngrok_auth(self, auth_token=None):
        """Setup ngrok authentication token"""
        if auth_token:
            print("🔐 Setting up ngrok authentication...")
            try:
                from pyngrok import conf, ngrok
                conf.get_default().auth_token = auth_token
                print("✅ ngrok authentication configured")
                return True
            except Exception as e:
                print(f"⚠️ Error setting auth token: {e}")
                print("You can set the auth token manually with: ngrok authtoken YOUR_TOKEN")
                return False
        else:
            print("ℹ️ No auth token provided. Using free tier (limited sessions)")
            return True
    
    def start_streamlit_app(self, app_file="2.8-Streamlit-Web-UI-Development.py"):
        """Start the Streamlit application"""
        print(f"🚀 Starting Streamlit app: {app_file}")
        
        if not os.path.exists(app_file):
            print(f"❌ Streamlit app file not found: {app_file}")
            return False
        
        try:
            # Start Streamlit in a separate process
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", app_file,
                "--server.headless", "true",
                "--server.port", "8501",
                "--browser.gatherUsageStats", "false"
            ])
            
            # Wait for Streamlit to start
            print("⏳ Waiting for Streamlit to start...")
            time.sleep(5)
            
            # Check if Streamlit is running
            for i in range(10):
                try:
                    response = requests.get("http://localhost:8501", timeout=2)
                    if response.status_code == 200:
                        print("✅ Streamlit app is running on http://localhost:8501")
                        return True
                except requests.RequestException:
                    time.sleep(2)
            
            print("❌ Streamlit app failed to start properly")
            return False
            
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def create_ngrok_tunnel(self, port=8501):
        """Create ngrok tunnel to expose Streamlit app"""
        print("🌐 Creating ngrok tunnel...")
        
        try:
            from pyngrok import ngrok
            
            # Create tunnel
            public_url = ngrok.connect(port)
            self.public_url = str(public_url)
            
            print(f"✅ ngrok tunnel created successfully!")
            print(f"🌍 Public URL: {self.public_url}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating ngrok tunnel: {e}")
            print("Make sure ngrok is properly installed and authenticated")
            return False
    
    def save_deployment_info(self):
        """Save deployment information to file"""
        deployment_info = {
            "deployment_time": datetime.now().isoformat(),
            "public_url": self.public_url,
            "local_url": "http://localhost:8501",
            "status": "active",
            "app_file": "2.8-Streamlit-Web-UI-Development.py"
        }
        
        with open("deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        print("💾 Deployment info saved to deployment_info.json")
    
    def display_access_info(self):
        """Display access information for the deployed app"""
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print(f"🌍 Public URL: {self.public_url}")
        print(f"🏠 Local URL:  http://localhost:8501")
        print("="*60)
        print("📋 Access Instructions:")
        print("1. Share the Public URL with others for remote access")
        print("2. Use Local URL for testing on your machine") 
        print("3. Keep this script running to maintain the tunnel")
        print("4. Press Ctrl+C to stop the deployment")
        print("="*60)
    
    def monitor_deployment(self):
        """Monitor the deployment and keep it running"""
        print("\n🔄 Monitoring deployment... (Press Ctrl+C to stop)")
        
        try:
            while True:
                # Check if Streamlit is still running
                if self.streamlit_process.poll() is not None:
                    print("❌ Streamlit process stopped")
                    break
                
                # Check if ngrok tunnel is still active
                try:
                    response = requests.get(self.public_url, timeout=10)
                    if response.status_code != 200:
                        print("⚠️ App may not be responding properly")
                except:
                    print("⚠️ Cannot reach the public URL")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Deployment stopped by user")
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes and resources"""
        print("🧹 Cleaning up...")
        
        # Stop ngrok tunnel
        try:
            from pyngrok import ngrok
            ngrok.kill()
            print("✅ ngrok tunnel closed")
        except:
            pass
        
        # Stop Streamlit process
        if self.streamlit_process:
            self.streamlit_process.terminate()
            self.streamlit_process.wait()
            print("✅ Streamlit process stopped")
    
    def deploy(self, auth_token=None, app_file="2.8-Streamlit-Web-UI-Development.py"):
        """Main deployment method"""
        print("🚀 Starting Heart Disease Prediction App Deployment")
        print("="*60)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed")
            return False
        
        # Step 2: Install ngrok
        if not self.install_ngrok():
            print("❌ ngrok setup failed")
            return False
        
        # Step 3: Setup authentication
        self.setup_ngrok_auth(auth_token)
        
        # Step 4: Start Streamlit app
        if not self.start_streamlit_app(app_file):
            print("❌ Streamlit startup failed")
            return False
        
        # Step 5: Create ngrok tunnel
        if not self.create_ngrok_tunnel():
            print("❌ ngrok tunnel creation failed")
            self.cleanup()
            return False
        
        # Step 6: Save deployment info
        self.save_deployment_info()
        
        # Step 7: Display access information
        self.display_access_info()
        
        # Step 8: Monitor deployment
        self.monitor_deployment()
        
        return True

def create_deployment_script():
    """Create a standalone deployment script"""
    script_content = '''#!/usr/bin/env python3
"""
Heart Disease Prediction App - Quick Deploy Script
Run this script to deploy the Streamlit app with ngrok
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deployment_module import NgrokDeployer

def main():
    """Quick deployment with user input"""
    print("❤️ Heart Disease Prediction App - Quick Deploy")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "2.8-Streamlit-Web-UI-Development.py",
        "final_heart_disease_model.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\\nPlease ensure all files are in the same directory.")
        return
    
    # Get ngrok auth token (optional)
    print("\\n🔐 ngrok Authentication (Optional)")
    print("Get free auth token at: https://dashboard.ngrok.com/get-started/your-authtoken")
    auth_token = input("Enter ngrok auth token (or press Enter to skip): ").strip()
    
    if not auth_token:
        auth_token = None
        print("ℹ️ Using free tier (limited to 2 hours)")
    
    # Start deployment
    deployer = NgrokDeployer()
    success = deployer.deploy(auth_token=auth_token)
    
    if not success:
        print("❌ Deployment failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("quick_deploy.py", "w") as f:
        f.write(script_content)
    
    print("✅ Quick deploy script created: quick_deploy.py")

def create_deployment_documentation():
    """Create comprehensive deployment documentation"""
    docs = """
# Heart Disease Prediction App - Deployment Guide

## Overview
This guide explains how to deploy the Heart Disease Prediction Streamlit application using ngrok for public access.

## Prerequisites

### Required Files
- `2.8-Streamlit-Web-UI-Development.py` - Main Streamlit application
- `final_heart_disease_model.pkl` - Trained ML model
- `2.9-Deployment-using-Ngrok.py` - This deployment script

### Required Python Packages
```bash
pip install streamlit plotly pandas numpy joblib scikit-learn pyngrok
```

## Deployment Methods

### Method 1: Using the Deployment Class

```python
from deployment_module import NgrokDeployer

# Create deployer instance
deployer = NgrokDeployer()

# Deploy with optional auth token
deployer.deploy(auth_token="your_ngrok_token_here")
```

### Method 2: Quick Deploy Script

```bash
python quick_deploy.py
```

### Method 3: Manual Deployment

1. **Start Streamlit App:**
   ```bash
   streamlit run "2.8-Streamlit-Web-UI-Development.py" --server.port 8501
   ```

2. **In a separate terminal, start ngrok:**
   ```bash
   ngrok http 8501
   ```

3. **Copy the public URL from ngrok output**

## ngrok Setup

### 1. Sign up for ngrok (Free)
- Go to [https://ngrok.com/](https://ngrok.com/)
- Create a free account
- Get your auth token from the dashboard

### 2. Install ngrok
```bash
# Option 1: Install via pip (recommended)
pip install pyngrok

# Option 2: Download binary
# Go to https://ngrok.com/download
```

### 3. Authenticate (Optional but Recommended)
```bash
ngrok authtoken YOUR_AUTH_TOKEN_HERE
```

## Usage Instructions

### For Developers
1. Run the deployment script
2. Wait for the public URL to be generated
3. Share the URL with users
4. Keep the script running to maintain access

### For End Users
1. Access the shared public URL
2. Fill in patient information in the web interface
3. Click "Make Prediction" to get results
4. Explore data visualizations in the Data Explorer tab

## Features of the Web Application

### 🏠 Prediction Page
- **Interactive Form**: User-friendly input form for patient data
- **Real-time Predictions**: Instant heart disease risk assessment
- **Probability Visualization**: Clear probability charts
- **Medical Disclaimers**: Appropriate medical advice warnings

### 📊 Data Explorer
- **Demographics Analysis**: Age and gender distributions
- **Clinical Features**: Chest pain types, blood pressure vs cholesterol
- **Correlation Matrix**: Feature relationships visualization
- **Statistics Dashboard**: Dataset overview and metrics

### ℹ️ About Page
- **Model Information**: Algorithm details and performance metrics
- **Technical Stack**: Technologies used in the project
- **Dataset Information**: UCI Heart Disease dataset details

## Security and Privacy

### Data Handling
- **No Data Storage**: Input data is not permanently stored
- **Session-based**: Each prediction is independent
- **Privacy First**: No personal information is logged

### Deployment Security
- **HTTPS by Default**: ngrok provides secure tunnels
- **Temporary URLs**: URLs expire when tunnel is closed
- **Rate Limiting**: Built-in ngrok rate limits for free tier

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Ensure `final_heart_disease_model.pkl` is in the same directory
   - Run the ML pipeline first to generate the model

2. **"ngrok tunnel failed"**
   - Check internet connection
   - Verify ngrok authentication token
   - Try restarting the deployment

3. **"Streamlit won't start"**
   - Check if port 8501 is already in use
   - Install missing dependencies
   - Verify Python environment

4. **"Public URL not accessible"**
   - Check ngrok tunnel status
   - Verify firewall settings
   - Try refreshing the URL

### Performance Tips
- **Free Tier Limitations**: 2-hour session limit, reconnect as needed
- **Paid Plan Benefits**: Custom domains, no time limits, more tunnels
- **Local Testing**: Always test locally before sharing publicly

## Monitoring and Maintenance

### Deployment Monitoring
- The script automatically monitors both Streamlit and ngrok processes
- Checks connectivity every 30 seconds
- Provides status updates and error notifications

### Log Files
- `deployment_info.json`: Contains deployment details and public URL
- Streamlit logs: Available in the terminal output
- ngrok logs: Available through ngrok dashboard

## Advanced Configuration

### Custom Port
```python
deployer.create_ngrok_tunnel(port=8502)  # Use different port
```

### Custom Streamlit Settings
```python
deployer.start_streamlit_app(app_file="your_app.py")
```

### Environment Variables
```bash
export NGROK_AUTHTOKEN=your_token_here
export STREAMLIT_SERVER_PORT=8501
```

## Cost Considerations

### Free Tier (ngrok)
- 2-hour session limit
- Random subdomain URLs
- Basic traffic monitoring

### Paid Plans (from $8/month)
- No session limits
- Custom domains
- Advanced features
- Better performance

## Support and Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ngrok Documentation](https://ngrok.com/docs/)
- [Project GitHub Repository](https://github.com/your-repo)

### Community Support
- Streamlit Community Forum
- Stack Overflow (tag: streamlit, ngrok)
- GitHub Issues

## Conclusion

This deployment solution provides an easy way to share your Heart Disease Prediction application with others. The combination of Streamlit and ngrok offers a professional, accessible interface for machine learning model demonstration and testing.

For production deployment, consider using cloud platforms like Heroku, AWS, or Google Cloud for better scalability and reliability.
"""
    
    with open("DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(docs)
    
    print("✅ Deployment documentation created: DEPLOYMENT_GUIDE.md")

def main():
    """Main function to demonstrate deployment"""
    print("🚀 Heart Disease Prediction - ngrok Deployment Setup")
    print("="*60)
    
    # Create deployment utilities
    create_deployment_script()
    create_deployment_documentation()
    
    print("\n📁 Deployment files created:")
    print("  - 2.9-Deployment-using-Ngrok.py (this file)")
    print("  - quick_deploy.py (easy deployment script)")
    print("  - DEPLOYMENT_GUIDE.md (comprehensive documentation)")
    
    print("\n🎯 Next Steps:")
    print("1. Ensure you have completed Tasks 2.1-2.7 first")
    print("2. Make sure 'final_heart_disease_model.pkl' exists")
    print("3. Run: python quick_deploy.py")
    print("4. Share the generated public URL")
    
    print("\n💡 Quick Start:")
    print("python quick_deploy.py")

if __name__ == "__main__":
    main()