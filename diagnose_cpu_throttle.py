"""
CPU 降频诊断脚本

用于诊断 SenseVoice CPU 推理时的降频问题
"""
import psutil
import time
import multiprocessing
import platform

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"CPU 型号: {platform.processor()}")
    print(f"物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"逻辑核心数: {psutil.cpu_count(logical=True)}")
    print(f"总内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print()

def monitor_cpu_frequency(duration=10):
    """监控 CPU 频率"""
    print("=" * 60)
    print(f"监控 CPU 频率（{duration}秒）")
    print("=" * 60)

    frequencies = []
    for i in range(duration):
        freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)

        print(f"[{i+1:2d}s] 当前频率: {freq.current:7.2f} MHz | "
              f"最小: {freq.min:7.2f} MHz | "
              f"最大: {freq.max:7.2f} MHz | "
              f"CPU使用率: {cpu_percent:5.1f}%")

        frequencies.append(freq.current)

    avg_freq = sum(frequencies) / len(frequencies)
    min_freq = min(frequencies)
    max_freq = max(frequencies)

    print()
    print(f"平均频率: {avg_freq:.2f} MHz")
    print(f"最低频率: {min_freq:.2f} MHz")
    print(f"最高频率: {max_freq:.2f} MHz")
    print()

    # 判断是否降频
    if min_freq < 2500:
        print("⚠️  检测到 CPU 降频！")
        print(f"   最低频率 {min_freq:.0f} MHz 低于 2500 MHz")
    else:
        print("✓  CPU 频率正常")
    print()

def test_onnx_cpu_load():
    """测试 ONNX Runtime CPU 负载"""
    print("=" * 60)
    print("测试 ONNX Runtime CPU 负载")
    print("=" * 60)

    try:
        import onnxruntime as ort
        import numpy as np

        print(f"ONNX Runtime 版本: {ort.__version__}")
        print(f"可用提供者: {ort.get_available_providers()}")
        print()

        # 创建一个简单的测试会话
        print("创建测试会话（模拟 SenseVoice 配置）...")

        total_cores = multiprocessing.cpu_count()
        reserved_cores = 4
        intra_op_threads = max(1, total_cores - reserved_cores)

        print(f"总核心数: {total_cores}")
        print(f"预留核心: {reserved_cores}")
        print(f"ONNX 使用核心: {intra_op_threads}")
        print()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = 1

        print("✓  ONNX Runtime 配置成功")
        print()

        # 建议
        print("建议的线程配置：")
        print(f"  - 当前配置: {intra_op_threads} 线程")

        if intra_op_threads > 16:
            print(f"  ⚠️  线程数过多可能导致调度开销")
            print(f"  建议: 尝试减少到 8-12 线程")
        elif intra_op_threads > 12:
            print(f"  ⚠️  线程数较多，可能触发功耗墙")
            print(f"  建议: 尝试减少到 8 线程")
        else:
            print(f"  ✓  线程数合理")
        print()

    except ImportError:
        print("❌ ONNX Runtime 未安装")
        print("   请运行: pip install onnxruntime")
        print()

def check_power_settings():
    """检查电源设置"""
    print("=" * 60)
    print("电源设置检查")
    print("=" * 60)

    if platform.system() == "Windows":
        print("请手动检查以下设置：")
        print()
        print("1. Windows 电源模式")
        print("   设置 -> 系统 -> 电源 -> 电源模式")
        print("   建议: 最佳性能")
        print()
        print("2. 高级电源设置")
        print("   控制面板 -> 电源选项 -> 更改计划设置 -> 更改高级电源设置")
        print("   - 处理器电源管理 -> 最小处理器状态: 100%")
        print("   - 处理器电源管理 -> 最大处理器状态: 100%")
        print()
        print("3. BIOS 设置")
        print("   重启进入 BIOS，检查：")
        print("   - CPU Power Limit (PL1/PL2)")
        print("   - Turbo Boost 是否启用")
        print("   - 散热模式设置")
        print()
    else:
        print("非 Windows 系统，请手动检查电源管理设置")
        print()

def main():
    """主函数"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "CPU 降频诊断工具" + " " * 25 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. 打印系统信息
    print_system_info()

    # 2. 监控 CPU 频率
    print("准备监控 CPU 频率...")
    print("提示: 如果要测试 SenseVoice 运行时的频率，请在另一个终端启动转录任务")
    print()
    input("按 Enter 开始监控...")
    print()

    monitor_cpu_frequency(duration=10)

    # 3. 测试 ONNX 配置
    test_onnx_cpu_load()

    # 4. 检查电源设置
    check_power_settings()

    print("=" * 60)
    print("诊断完成")
    print("=" * 60)
    print()
    print("如果检测到降频，请按以下顺序排查：")
    print("1. 检查 CPU 温度（用 HWiNFO64）")
    print("2. 检查电源模式（设置为最佳性能）")
    print("3. 尝试减少 ONNX 线程数（修改代码）")
    print("4. 检查 BIOS 功耗限制")
    print("5. 清理散热模组/更换硅脂")
    print()

if __name__ == "__main__":
    main()
