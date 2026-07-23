%DEPLOYDETECTORJETSON Genera y despliega el nodo ROS 2 de detección.
%
%   GPU Coder + ROS Toolbox generan el nodo ROS 2 completo a partir de
%   detectorJetsonNode.m, transfieren el código a la Jetson, lo compilan
%   allí con colcon (compilación remota: no se necesita CUDA en el host)
%   y lo arrancan.
%
%   Flujo del script:
%       1. Prepara 'detectorDeploy.mat' (solo el detector) a partir de
%          trainedDetector.mat, requisito de coder.loadDeepLearningNetwork.
%       2. (Opcional) Comprueba el entorno GPU de la Jetson por SSH.
%       3. Configura GPU Coder: destino ROS 2 remoto, GPU Orin (compute
%          capability 8.7) y librería de inferencia.
%       4. Genera, compila en la Jetson y arranca el nodo.
%
%   Requisitos (host): MATLAB Coder, GPU Coder, ROS Toolbox, add-on
%       "GPU Coder Interface for Deep Learning" y, para la verificación
%       opcional, "MATLAB Coder Support Package for NVIDIA Jetson and
%       NVIDIA DRIVE Platforms".
%   Requisitos (Jetson): JetPack 7.2 (Ubuntu 24.04, CUDA 13), ROS 2
%       Jazzy, workspace colcon y driver de cámara v4l2_camera
%       (ver README_ROS_OD.md).
%
%   Ver también: detectorJetsonNode, viewDetectorROS2

clear; clc;

%% 1. Parámetros de usuario ================================================
modelFile = "trainedDetector.mat";     % salida de trainDetector.m

jetsonIP       = "192.168.1.50";       % IP de la Jetson (editar)
jetsonUser     = "jetson";
jetsonPassword = "jetson";

ros2Folder    = "/opt/ros/jazzy";      % instalación de ROS 2 en la Jetson
                                       % (Jazzy en JetPack 7.2/Ubuntu 24.04)
ros2Workspace = "~/detector_ws";       % workspace colcon en la Jetson

runGpuPrecheck = true;
buildAction    = "Build and run";      % "None" | "Build and load" | "Build and run"

% Librería de inferencia en la GPU.
%   "cudnn"    : ruta VALIDADA con JetPack 7.2 (FP32).
%   "tensorrt" : mayor rendimiento (FP16). En la suite de clasificación
%                falló con JetPack 7.2 porque TensorRT 10.16 eliminó la
%                API de la capa fully-connected que R2025b aún usa
%                ('identifier "IFullyConnectedLayer" is undefined').
%                tiny-YOLOv4 NO tiene capas fully-connected, así que
%                "tensorrt" PODRÍA compilar aquí y vale la pena el
%                experimento; si falla, volver a "cudnn".
inferenceLibrary = "cudnn";            % "cudnn" | "tensorrt"

%% 2. Preparar el MAT de despliegue =======================================
assert(isfile(modelFile), "No se encontró el modelo: %s", modelFile);
S = load(modelFile, "detector", "inputSize");

% coder.loadDeepLearningNetwork exige un MAT con un ÚNICO objeto de red.
detector = S.detector; %#ok<NASGU>
save("detectorDeploy.mat", "detector");
fprintf("Preparado detectorDeploy.mat a partir de '%s'.\n", modelFile);
fprintf("Entrada del detector: [%d %d %d]\n\n", S.inputSize);

%% 3. (Opcional) Verificación del entorno GPU de la Jetson ================
if runGpuPrecheck
    try
        fprintf("Verificando entorno GPU de la Jetson (%s)...\n", jetsonIP);
        hw = jetson(jetsonIP, jetsonUser, jetsonPassword);

        envCfg                = coder.gpuEnvConfig("jetson");
        envCfg.DeepLibTarget  = char(inferenceLibrary);
        % El sub-test de compilación de checkGpuInstall usa la
        % arquitectura GPU por defecto (sm_50), que el nvcc de CUDA 13
        % ya no acepta y no es configurable aquí. Se desactiva: este
        % precheck solo verifica la presencia de CUDA/cuDNN/TensorRT, y
        % la compilación real (sección 4, con ComputeCapability 8.7)
        % actúa como prueba definitiva.
        envCfg.DeepCodegen    = 0;
        envCfg.Quiet          = 1;
        envCfg.HardwareObject = hw;
        coder.checkGpuInstall(envCfg);
        fprintf("Entorno GPU verificado correctamente.\n\n");
    catch ME
        % NOTA: concatenar con comillas simples (char), no con comillas
        % dobles: ["a","b"] crea un string array y warning lo rechaza.
        warning(['Verificación GPU omitida o fallida (no bloquea el ' ...
            'despliegue). Detalle: %s'], ME.message);
    end
end

%% 4. Configuración de GPU Coder para nodo ROS 2 remoto ===================
cfgGen = coder.gpuConfig("exe");

% GPU de destino: la Orin Nano (arquitectura Ampere) tiene compute
% capability 8.7. Es IMPRESCINDIBLE declararla: el valor por defecto de
% GPU Coder (sm_50) ya no es aceptado por el nvcc de CUDA 13 ("nvcc
% fatal: Unsupported gpu architecture 'sm_50'").
cfgGen.GpuConfig.ComputeCapability = "8.7";

cfgGen.Hardware = coder.hardware("Robot Operating System 2 (ROS 2)");
cfgGen.Hardware.BuildAction          = buildAction;
cfgGen.Hardware.DeployTo             = "Remote Device";
cfgGen.Hardware.RemoteDeviceAddress  = jetsonIP;
cfgGen.Hardware.RemoteDeviceUsername = jetsonUser;
cfgGen.Hardware.RemoteDevicePassword = jetsonPassword;
cfgGen.Hardware.ROS2Folder           = ros2Folder;
cfgGen.Hardware.ROS2Workspace        = ros2Workspace;

% FP16 solo con TensorRT (~2x rendimiento en la Orin); cuDNN usa FP32.
dlcfg = coder.DeepLearningConfig(inferenceLibrary);
if inferenceLibrary == "tensorrt"
    dlcfg.DataType = "fp16";
end
cfgGen.DeepLearningConfig = dlcfg;

%% 5. Generación de código y despliegue ===================================
fprintf("Generando código CUDA y compilando en la Jetson (colcon)...\n");
fprintf("Este proceso puede tardar varios minutos.\n\n");

codegen detectorJetsonNode -config cfgGen

%% 6. Instrucciones posteriores ===========================================
fprintf("\n================= DESPLIEGUE COMPLETADO =================\n");
fprintf("El nodo quedó corriendo en la Jetson (BuildAction='Build and run').\n");
fprintf("IMPORTANTE: el nodo solo publica cuando recibe imágenes; el driver\n");
fprintf("de cámara (v4l2_camera) debe estar publicando en /image_raw.\n\n");
fprintf("Comprobaciones útiles (en la Jetson):\n");
fprintf("  ros2 node list                          -> debe aparecer /detector_jetson\n");
fprintf("  ros2 topic hz /detector/detections      -> ~20-30 Hz con la cámara activa\n\n");
fprintf("Para relanzar el nodo manualmente (p. ej. tras un reinicio):\n");
fprintf("  source %s/setup.bash\n", ros2Folder);
fprintf("  source %s/install/setup.bash\n", ros2Workspace);
fprintf("  ros2 run detectorjetsonnode detectorJetsonNode\n");
fprintf("  (el ejecutable conserva las mayúsculas del entry-point; verificar\n");
fprintf("   el nombre exacto con: ros2 pkg executables detectorjetsonnode)\n");
fprintf("  Evitar instancias duplicadas: si /detector_jetson ya existe,\n");
fprintf("  terminar con: pkill -f detectorJetsonNode\n\n");
if inferenceLibrary == "tensorrt"
    fprintf("El primer arranque tarda 1-3 min (TensorRT optimiza el motor).\n");
else
    fprintf("Con cuDNN el arranque del nodo es casi inmediato.\n");
end
fprintf("Luego, en el host, ejecute viewDetectorROS2.m\n");
