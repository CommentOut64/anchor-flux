# -*- coding: utf-8 -*-
"""
CPU亲和性功能快速验证脚本
"""

import sys
import os

# 添加路径以导入处理器
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app'))

try:
    from processor import CPUAffinityManager, CPUAffinityConfig
    
    def quick_test():
        """快速测试CPU亲和性功能"""
        print("Starting CPU Affinity Quick Test...")
        
        manager = CPUAffinityManager()
        sys_info = manager.get_system_info()
        
        print(f"CPU Support: {sys_info.get('supported', False)}")
        print(f"Logical Cores: {sys_info.get('logical_cores', 'Unknown')}")
        print(f"Physical Cores: {sys_info.get('physical_cores', 'Unknown')}")
        print(f"Platform: {sys_info.get('platform', 'Unknown')}")
        
        if sys_info.get('supported', False):
            # 测试auto策略
            auto_cores = manager.calculate_optimal_cores(strategy="auto")
            print(f"Auto Strategy Cores: {auto_cores}")
            
            # 测试half策略
            half_cores = manager.calculate_optimal_cores(strategy="half")
            print(f"Half Strategy Cores: {half_cores}")
            
            # 测试应用CPU亲和性
            config = CPUAffinityConfig(enabled=True, strategy="half")
            success = manager.apply_cpu_affinity(config)
            print(f"Apply CPU Affinity: {success}")
            
            if success:
                try:
                    import psutil
                    current_affinity = psutil.Process().cpu_affinity()
                    print(f"Current Affinity After Apply: {current_affinity}")
                    
                    # 恢复原始设置
                    restored = manager.restore_cpu_affinity()
                    print(f"Restore Affinity: {restored}")
                    
                    if restored:
                        current_affinity = psutil.Process().cpu_affinity()
                        print(f"Current Affinity After Restore: {len(current_affinity)} cores")
                        
                except Exception as e:
                    print(f"Error checking affinity: {e}")
        
        print("CPU Affinity Quick Test Completed!")
        return True
        
    if __name__ == "__main__":
        quick_test()
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Test error: {e}")