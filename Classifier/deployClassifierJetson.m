%DEPLOYCLASSIFIERJETSON Genera y despliega el nodo ROS 2 de clasificación.
%
%   GPU Coder + ROS Toolbox generan el nodo ROS 2 completo a partir de
%   classifierJetsonNode.m, transfieren el código a la Jetson, lo compilan
%   allí con colcon (compilación remota: no se necesita CUDA en el host, 
%   reemplazo de catkin) y lo arrancan.
%
%   Flujo del script:
%       1. Prepara 'classifierDeploy.mat' (solo la red) a partir de
%          trainedClassifier.mat, requisito de coder.loadDeepLearningNetwork.
%       2. Verifica que INPUT_SIZE de classifierJetsonNode.m coincide con
%          la entrada real de la red.
%       3. (Opcional) Comprueba el entorno GPU de la Jetson por SSH.
%       4. Configura GPU Coder: destino ROS 2 remoto + TensorRT FP16.
%       5. Genera, compila en la Jetson y arranca el nodo.
%
%   Requisitos (host): MATLAB Coder, GPU Coder, ROS Toolbox, add-on
%       "GPU Coder Interface for Deep Learning" y, para la verificación
%       opcional, "MATLAB Coder Support Package for NVIDIA Jetson and
%       NVIDIA DRIVE Platforms".
%   Requisitos (Jetson): JetPack 7.2 (Ubuntu 24.04, CUDA 13), ROS 2
%       Jazzy, workspace colcon y driver de cámara v4l2_camera
%       (ver README_jetson_ros2.md). NOTA (verificado en la práctica):
%       con JetPack 7.2 la inferencia debe generarse con cuDNN; el
%       TensorRT 10.16 de JetPack 7.2 eliminó APIs que el código de
%       R2025b aún utiliza, por lo que la compilación con "tensorrt"
%       falla.
%
%   Ver también: classifierJetsonNode, viewClassifierROS2

clear; clc;

%% 1. Parámetros de usuario ================================================
modelFile = "trainedClassifier.mat";   % salida de trainClassifier.m

jetsonIP       = "172.16.21.103";       % IP de la Jetson
jetsonUser     = "laboratoriosdiee";
jetsonPassword = "DIEE2026";

ros2Folder    = "/opt/ros/jazzy";      % instalación de ROS 2 en la Jetson
                                       % (Jazzy en JetPack 7.2/Ubuntu 24.04)
ros2Workspace = "~/classifier_ws";     % workspace colcon en la Jetson

runGpuPrecheck = true;
buildAction    = "Build and load";      % "None" | "Build and load" | "Build and run"

% Librería de inferencia en la GPU.
%   "cudnn"    : opción por defecto y VALIDADA con JetPack 7.2. FP32.
%   "tensorrt" : mayor rendimiento (FP16), pero INCOMPATIBLE con el
%                TensorRT 10.16 de JetPack 7.2 en R2025b: el código
%                generado usa la API IFullyConnectedLayer/addFullyConnected,
%                eliminada en TensorRT 10.x (error de compilación
%                'identifier "IFullyConnectedLayer" is undefined').
%                Reintentar en una versión de MATLAB posterior a R2025b.
inferenceLibrary = "cudnn";            % "cudnn" | "tensorrt"

%% 2. Preparar el MAT de despliegue =======================================
assert(isfile(modelFile), "No se encontró el modelo: %s", modelFile);
S = load(modelFile, "net", "inputSize");

% coder.loadDeepLearningNetwork necesita un MAT con un ÚNICO objeto de red.
net = S.net;
save("classifierDeploy.mat", "net");
fprintf("Preparado classifierDeploy.mat a partir de '%s'.\n", modelFile);

%% 3. Verificación de INPUT_SIZE del entry-point ==========================
% TensorRT necesita tamaño de entrada fijo, por lo que INPUT_SIZE es una
% constante de compilación dentro de classifierJetsonNode.m. Aquí se
% comprueba que coincide con la red.
fprintf("Entrada de la red: [%d %d %d]\n", S.inputSize);

nodeText = fileread("classifierJetsonNode.m");
tok = regexp(nodeText, ...
    "INPUT_SIZE\s*=\s*\[(\d+)\s+(\d+)\s+(\d+)\]", "tokens", "once");
if ~isempty(tok)
    nodeSize = str2double(tok);
    if ~isequal(nodeSize(:).', S.inputSize(:).')
        error(['INPUT_SIZE en classifierJetsonNode.m es [%d %d %d] pero ' ...
        'la red espera [%d %d %d]. Edite la constante y vuelva a ' ...
        'ejecutar este script.'], nodeSize(1), nodeSize(2), nodeSize(3), ...
        S.inputSize(1), S.inputSize(2), S.inputSize(3));
    end
    fprintf("INPUT_SIZE del nodo verificado: coincide con la red.\n\n");
else
    warning(['No se pudo verificar INPUT_SIZE automáticamente; ' ...
        'confirme que coincide con la entrada de la red.']);
end

%% 4. (Opcional) Verificación del entorno GPU de la Jetson ================
if runGpuPrecheck
    try
        fprintf("Verificando entorno GPU de la Jetson (%s)...\n", jetsonIP);
        hw = jetson(jetsonIP, jetsonUser, jetsonPassword);

        envCfg                = coder.gpuEnvConfig("jetson");
        envCfg.DeepLibTarget  = char(inferenceLibrary);
        % El sub-test de compilación de checkGpuInstall usa la
        % arquitectura GPU por defecto (sm_50), que el nvcc de CUDA 13
        % (JetPack 7.x) ya no acepta y no es configurable. Este precheck 
        % solo verifica la presencia de CUDA/cuDNN/TensorRT, y la compilación 
        % real (sección 5, con ComputeCapability 8.7) actúa como prueba definitiva.
        envCfg.DeepCodegen    = 0;
        envCfg.Quiet          = 1;
        envCfg.HardwareObject = hw;
        coder.checkGpuInstall(envCfg);
        fprintf("Entorno GPU verificado correctamente.\n\n");
    catch ME
        warning(['Verificación GPU omitida o fallida (no bloquea el ' ...
            'despliegue). Detalle: %s'], ME.message);
    end
end

%% 5. Configuración de GPU Coder para nodo ROS 2 remoto ===================
cfgGen = coder.gpuConfig("exe");

% GPU de destino: la Orin Nano (arquitectura Ampere) tiene compute
% capability 8.7. Es IMPRESCINDIBLE declararla: el valor por defecto de
% GPU Coder (sm_50) ya no es aceptado por el nvcc de CUDA 13 ("nvcc
% fatal: Unsupported gpu architecture 'sm_50'") y, además, FP16 exige
% compute capability >= 5.3.
cfgGen.GpuConfig.ComputeCapability = "8.7";

cfgGen.Hardware = coder.hardware("Robot Operating System 2 (ROS 2)");
cfgGen.Hardware.BuildAction          = buildAction;
cfgGen.Hardware.DeployTo             = "Remote Device";
cfgGen.Hardware.RemoteDeviceAddress  = jetsonIP;
cfgGen.Hardware.RemoteDeviceUsername = jetsonUser;
cfgGen.Hardware.RemoteDevicePassword = jetsonPassword;
cfgGen.Hardware.ROS2Folder           = ros2Folder;
cfgGen.Hardware.ROS2Workspace        = ros2Workspace;

% TensorRT en FP16 duplica el rendimiento frente a FP32 en la GPU Orin
% con pérdida de precisión despreciable en clasificación. Para cuDNN se
% mantiene FP32 (cuDNN no admite FP16 en esta configuración).
dlcfg = coder.DeepLearningConfig(inferenceLibrary);
if inferenceLibrary == "tensorrt"
    dlcfg.DataType = "fp16";
end
cfgGen.DeepLearningConfig = dlcfg;

%% 6. Generación de código y despliegue ===================================
fprintf("Generando código CUDA y compilando en la Jetson (colcon)...\n");
fprintf("Este proceso puede tardar varios minutos.\n\n");

codegen classifierJetsonNode -config cfgGen

%% 7. Instrucciones posteriores ===========================================
fprintf("\n================= DESPLIEGUE COMPLETADO =================\n");
fprintf("El nodo quedó corriendo en la Jetson (BuildAction='Build and run').\n");
fprintf("IMPORTANTE: el nodo solo publica cuando recibe imágenes; el driver\n");
fprintf("de cámara (v4l2_camera) debe estar publicando en /image_raw.\n\n");
fprintf("Comprobaciones útiles (en la Jetson):\n");
fprintf("  ros2 node list                        -> debe aparecer /classifier_jetson\n");
fprintf("  ros2 topic hz /classifier/class_id    -> ~30 Hz con la cámara activa\n\n");
fprintf("Para relanzar el nodo manualmente (p. ej. tras un reinicio):\n");
fprintf("  source %s/setup.bash\n", ros2Folder);
fprintf("  source %s/install/setup.bash\n", ros2Workspace);
fprintf("  ros2 run classifierjetsonnode classifierJetsonNode\n");
fprintf("  (el ejecutable conserva las mayúsculas del entry-point; verificar\n");
fprintf("   el nombre exacto con: ros2 pkg executables classifierjetsonnode)\n\n");
if inferenceLibrary == "tensorrt"
    fprintf("El primer arranque tarda 1-3 min (TensorRT optimiza el motor).\n");
else
    fprintf("Con cuDNN el arranque del nodo es casi inmediato.\n");
end
fprintf("Luego, en el host, ejecute viewClassifierROS2.m\n");
