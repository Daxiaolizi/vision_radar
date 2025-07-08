import sys
import os
import platform
os.environ['LD_LIBRARY_PATH'] = '/usr/local/TensorRT-8.5.1.7/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
tensorrt_python_path = '/usr/local/TensorRT-8.5.1.7/python'
sys.path.insert(0, tensorrt_python_path)  # 使用 insert(0) 确保优先级最高

# 调试信息 - 检查路径是否添加成功
print(f"System paths: {sys.path}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")

# 尝试提前导入 tensorrt 以验证
try:
    import tensorrt as trt
    print(f"Successfully imported TensorRT {trt.__version__}")
except ImportError as e:
    print(f"TensorRT import failed: {e}")
    # 尝试从常见位置导入
    try:
        from tensorrt import *
        print("Imported tensorrt from alternative path")
    except:
        print("All import attempts failed")
        raise



from export import export_onnx
from utils.general import (LOGGER, check_requirements, check_version,
                           colorstr)


def export_engine( file, half, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt

    check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0

    onnx = file + '.onnx'

    LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    # assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file + '.engine'  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # config.max_workspace_size = workspace * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')


    LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return f, None


export_engine('models/armor', half=True)
export_engine('models/car', half=True)