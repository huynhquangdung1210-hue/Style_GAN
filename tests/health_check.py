"""Simple health check for the style transfer system"""

import sys
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all key modules can be imported"""
    try:
        print("🧪 Testing imports...")
        
        # Test config
        from utils.config import get_settings
        print("✅ Config module imported")
        
        # Test storage 
        from utils.storage_simple import get_storage_client
        print("✅ Storage module imported")
        
        # Get settings
        settings = get_settings()
        print(f"✅ Settings loaded - Storage: {settings.storage.storage_type}")
        
        print("\n🎉 All core modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_storage():
    """Test storage connectivity"""
    try:
        print("\n🗄️ Testing storage...")
        from utils.storage_simple import get_storage_client
        
        client = get_storage_client()
        print("✅ Storage client created")
        
        # Simple connectivity test
        if hasattr(client, 'settings'):
            print(f"✅ Storage type: {client.settings.storage_type}")
            if client.settings.storage_type == "minio":
                print(f"✅ MinIO endpoint: {client.settings.minio_endpoint}")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False

def main():
    """Run system health checks"""
    print("🚀 Style Transfer System Health Check\n")
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test storage
        storage_ok = test_storage()
        
        if storage_ok:
            print("\n✅ System is ready!")
            print("\nNext steps:")
            print("1. Ensure MinIO is running: docker ps")
            print("2. Start API server: python run_server.py")
            print("3. Add test images to tests/test_images/")
            return True
    
    print("\n❌ System has issues that need to be resolved")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)