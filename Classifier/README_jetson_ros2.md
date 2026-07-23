# Configuración de la Jetson Orin Nano Super — ROS 2 Jazzy + cámara USB!!!

Guía paso a paso para configurar la **Jetson Orin Nano Super Developer Kit
(8 GB)** con **JetPack 7.2** lista para recibir nodos ROS 2 generados
desde MATLAB (GPU Coder) y publicar el video de una cámara USB en la
red ROS 2. Todos los comandos se ejecutan **en la Jetson** salvo que se
indique lo contrario. Seguir las secciones en orden: cada una incluye su
verificación.

## 0. Material necesario

- Jetson Orin Nano Super Developer Kit (8 GB) con su fuente de poder.
- Almacenamiento: NVMe o microSD de 64 GB o más (UHS-I U3 / A2).
- Cámara USB (UVC estándar) conectada a un puerto USB de la Jetson.
- PC host con MATLAB R2025b (Windows o Linux) y la Jetson en la
  **misma red**; se recomienda Ethernet (el video sin comprimir puede
  saturar WiFi).
- Monitor + teclado para el primer arranque (o configuración headless).

## 1. Sistema operativo: JetPack 7.2

JetPack 7.2 corresponde a Jetson Linux L4T r39.2: Ubuntu 24.04 LTS con
kernel 6.8 y CUDA 13. Ojito: **No** se instala Ubuntu por
separado.

1. Instalar **NVIDIA SDK Manager** (2.4.1 o superior) en un PC Linux y
   flashear JetPack 7.2 para *Jetson Orin Nano [8GB developer kit
   version]* con la Jetson en modo recovery (Direct Flash), o usar la
   imagen/instalador oficial de JetPack 7.2 si está disponible para
   microSD. Detalles en `https://developer.nvidia.com/embedded/jetpack`.
2. Arrancar y completar el asistente (usuario, contraseña, red).
3. Activar el modo de máximo rendimiento (MAXN SUPER):

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**Verificación:**

```bash
cat /etc/nv_tegra_release      # debe indicar R39 (revision 2.x)
```

> **Compatibilidad con MATLAB (verificado en R2025b).**
> Con JetPack 7.2 el flujo de despliegue FUNCIONA con estos ajustes ya
> incorporados en los scripts:
> 1) declarar la GPU real (`ComputeCapability = "8.7"`), porque el nvcc
> de CUDA 13 eliminó la arquitectura por defecto de GPU Coder (sm_50);
> 2) generar la inferencia con **cuDNN** (`inferenceLibrary = "cudnn"`):
> el TensorRT 10.16 de JetPack 7.2 eliminó APIs que el código de R2025b
> aún utiliza, así que la compilación con "tensorrt" falla. Reintentar
> TensorRT tras actualizar los support packages (Add-On Manager) o en
> versiones de MATLAB posteriores. Como alternativa, JetPack 6.2
> (Ubuntu 22.04 + ROS 2 Humble + TensorRT 10.3) sigue siendo la ruta
> validada oficialmente por MathWorks.

## 2. Componentes del SDK: CUDA, cuDNN y TensorRT

**Importante:** dependiendo del método de instalación, el flasheo puede
dejar solo el sistema operativo, **sin** CUDA/cuDNN/TensorRT (síntoma
típico: `nvcc: command not found` y ninguna carpeta `/usr/local/cuda*`).
El metapaquete `nvidia-jetpack` instala todos los componentes de una vez
(varios GB; tomará un buen tiempo):

```bash
sudo apt update
sudo apt install -y nvidia-jetpack
```

**Verificación:**

```bash
ls -d /usr/local/cuda*                          # debe existir /usr/local/cuda
dpkg -l | grep -E "cudnn|tensorrt" | head -5    # cuDNN y TensorRT instalados
```

Si existe la carpeta con versión (p. ej. `/usr/local/cuda-13.2`) pero
**no** el enlace `/usr/local/cuda`, crearlo:

```bash
sudo ln -s /usr/local/cuda-13.2 /usr/local/cuda   # ajustar a la versión real
```

## 3. Paquetes base del sistema

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git curl nano \
    v4l-utils libv4l-dev libsdl1.2debian libsdl1.2-dev
```

## 4. Variables de entorno CUDA (requisito de la compilación remota)

MATLAB compila el nodo por **SSH no interactivo**, y en Ubuntu el archivo
`~/.bashrc` termina temprano en sesiones no interactivas. Por eso las
variables deben insertarse **al inicio** del archivo. Ejecutar estos
comandos **una sola vez** (si se repiten, se insertan líneas duplicadas;
no rompe nada, pero conviene revisar con `head -5 ~/.bashrc` y borrar
repetidos con `nano ~/.bashrc`):

```bash
sed -i '1i export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' ~/.bashrc
sed -i '1i export PATH=/usr/local/cuda/bin:$PATH' ~/.bashrc
source ~/.bashrc
```

**Verificación en la Jetson:**

```bash
nvcc --version       # debe reportar release 13.2 (V13.2.x)
```

**Verificación desde el host:** esta es la prueba que realmente importa
de cara hacia MATLAB. Funciona igual desde PowerShell (Windows) o una terminal
Linux; sustituir usuario e IP. Debe imprimir la versión de CUDA sin abrir
sesión interactiva (la primera vez pedirá aceptar la huella del host con
`yes` y la contraseña):

```bash
ssh usuario@IP_JETSON 'nvcc --version'
```

## 5. Instalación de ROS 2 Jazzy

JetPack 7.2 está basado en Ubuntu 24.04 (noble) sobre arquitectura arm64,
cuya distribución ROS 2 LTS correspondiente es **Jazzy Jalisco**
(soportada hasta 2029). La línea del repositorio usa esos valores fijos
(`arm64`, `noble`): las sustituciones de shell `$( )` pueden
corromperse al copiar desde algunos visores de Markdown, que las
interpretan como ecuaciones.

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" | \
     sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

cat /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-jazzy-ros-base ros-jazzy-ament-cmake ros-dev-tools \
    python3-colcon-common-extensions

sed -i '1i source /opt/ros/jazzy/setup.bash' ~/.bashrc
source ~/.bashrc
```

El `cat` intermedio es una verificación (sanity check): debe mostrar **una sola línea**
que comience con `deb [arch=arm64 signed-by=...`. Si aparece cualquier
otra cosa (por ejemplo `%InlineEquation`), el comando se corrompió al
copiarlo: borrar el archivo con
`sudo rm /etc/apt/sources.list.d/ros2.list` y repetir el `echo`.

El `source` de ROS se inserta **al inicio** de `~/.bashrc` (con `sed`, una
sola vez) por la misma razón que las variables CUDA del paso 4: la
compilación remota que lanza MATLAB usa una sesión SSH **no interactiva**,
y Ubuntu corta el `.bashrc` temprano en esas sesiones. Si el `source`
queda al final, colcon corre sin el entorno ROS y CMake falla con
`Could not find ... "ament_cmake"`.

**Verificación en la Jetson** (debe listar `/parameter_events` y
`/rosout` sin errores):

```bash
ros2 topic list
```

**Verificación desde el host** (debe imprimir `jazzy`):

```bash
ssh usuario@IP_JETSON 'echo $ROS_DISTRO'
```

## 6. Workspace colcon

Carpeta donde MATLAB compilará el nodo generado. El nombre debe coincidir
con el parámetro `ros2Workspace` de `deployClassifierJetson.m`
(`~/classifier_ws` por defecto):

```bash
mkdir -p ~/classifier_ws/src
```

## 7. Dominio ROS 2

Todos los equipos de la misma aplicación deben compartir `ROS_DOMAIN_ID`
(0 por defecto). Se inserta al **inicio** de `~/.bashrc` (una sola vez)
para que también lo vea el nodo que MATLAB lanza por SSH no interactivo:

```bash
sed -i '1i export ROS_DOMAIN_ID=0' ~/.bashrc
source ~/.bashrc
```

Debe coincidir con el parámetro `domainId` de los scripts del host
(`viewCameraROS2.m`, `viewClassifierROS2.m`).

## 8. Cámara USB como nodo ROS 2

Instalar el driver y verificar que la cámara es detectada:

```bash
sudo apt install -y ros-jazzy-v4l2-camera ros-jazzy-image-transport-plugins
v4l2-ctl --list-devices
```

Lanzar el driver (dejar corriendo en una terminal dedicada):

```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p image_size:="[640,480]" -p output_encoding:="rgb8"
```

**Verificaciones** (en otra terminal):

```bash
ros2 topic list                                   # debe aparecer /image_raw
ros2 topic hz /image_raw                          # frecuencia de publicación
ros2 topic echo /image_raw --no-arr | head -n 8   # debe indicar encoding: rgb8
```

Notas:
- El nodo generado desde MATLAB asume `rgb8` en tiempo de compilación:
  no cambiar `output_encoding`.
- Si hay varias cámaras, seleccionar el dispositivo con
  `-p video_device:="/dev/video0"` (ver la lista de `v4l2-ctl`).
- **Mensajes normales que NO son errores** al lanzar el driver:
  `Permission denied (13)` en algún control de la cámara (peculiaridad
  UVC, no afecta la captura); `performing possibly slow conversion:
  yuv422_yuy2 => rgb8` (la cámara entrega YUYV y el driver lo convierte
  al rgb8 solicitado; a 640x480 mantiene ~30 fps); `Camera calibration
  file ... not found` (la calibración intrínseca no se usa en este
  proyecto). Asimismo, el `BrokenPipeError` de Python al final del
  comando `echo | head` es cosmético: `head` cierra la tubería tras
  mostrar las primeras líneas.

## 9. (Opcional) Arrancar la cámara automáticamente al encender

Servicio systemd para no depender de una terminal abierta. Sustituir
`jetson` por el usuario real en la línea `User=`:

```bash
sudo tee /etc/systemd/system/ros2-camera.service > /dev/null << 'EOF'
[Unit]
Description=Camara USB como nodo ROS 2 (v4l2_camera)
After=network.target

[Service]
User=jetson
Environment=ROS_DOMAIN_ID=0
ExecStart=/bin/bash -c 'source /opt/ros/jazzy/setup.bash && ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[640,480]" -p output_encoding:="rgb8"'
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ros2-camera.service
systemctl status ros2-camera.service
```

## 10. Verificación host ↔ Jetson

Con el driver de cámara corriendo, en **MATLAB (host)**:

```matlab
setenv("ROS_DOMAIN_ID", "0")
n = ros2node("/prueba");
ros2 topic list
```

Debe listar `/image_raw`. Para ver el video: ejecutar `viewCameraROS2.m`.
A partir de aquí la Jetson está lista para `deployClassifierJetson.m`.

## 11. Solución de problemas

| Síntoma | Causa probable / solución |
|---|---|
| `Could not find a package configuration file provided by "ament_cmake"` (a veces junto al warning de CMake "ROS 2 distribution is not recommended") | El entorno ROS 2 no es visible en la sesión SSH no interactiva donde MATLAB lanza colcon: mover el `source /opt/ros/jazzy/setup.bash` al **inicio** de `~/.bashrc` (paso 5) e instalar `ros-jazzy-ament-cmake`. Verificar desde el host: `ssh usuario@IP 'echo $ROS_DISTRO'` debe imprimir `jazzy`. |
| `ros2 run classifierjetsonnode classifierjetsonnode` → "No executable found" | El ejecutable conserva las mayúsculas del entry-point: `ros2 run classifierjetsonnode classifierJetsonNode`. Listar los ejecutables reales con `ros2 pkg executables classifierjetsonnode`. |
| `ros2 node list` avisa de nodos con nombre repetido (`/classifier_jetson` aparece dos veces) y los tópicos publican al doble de frecuencia | Hay dos instancias del nodo (la que dejó el despliegue con "Build and run" más una lanzada a mano). Terminar todas con `pkill -f classifierJetsonNode` y lanzar una sola. |
| En el host, `rosReadImage` falla con `Expected input to be nonempty` sobre `/classifier/image_view` | El código C++ generado puede no rellenar el campo `encoding` del mensaje. El visor (`viewClassifierROS2.m`) lo restituye a `rgb8` automáticamente, y el nodo lo asigna de forma explícita; verificar que ambos archivos están actualizados. |
| Warnings `Lost messages on topic ...` en el host | Benignos: la cola del suscriptor descarta cuadros cuando MATLAB procesa/dibuja más lento de lo que llegan. Se reducen al eliminar nodos duplicados. |
| `ros2 topic hz /classifier/class_id` no reporta nada aunque el nodo aparece en `ros2 node list` | El nodo solo publica cuando recibe imágenes: verificar que el driver de cámara está activo (`ros2 topic hz /image_raw` debe dar ~30 Hz). |
| Mensajes benignos del despliegue: `install/setup.bash: No such file or directory` (solo en el primer build; colcon crea ese archivo al terminar), `Function '...' does not terminate because of an infinite loop` (el `while true` del nodo es intencional: es su bucle principal) y `nvlink warning: SM Arch ('sm_75') not found` (cosmética del enlazador; los objetos son sm_87) | No requieren acción. |
| `error: identifier "IFullyConnectedLayer" is undefined` / `has no member "addFullyConnected"` al compilar en la Jetson | Incompatibilidad confirmada: TensorRT 10.16 (JetPack 7.2) eliminó esas APIs y el código generado por R2025b aún las usa. Usar `inferenceLibrary = "cudnn"` en el script de despliegue (valor por defecto). Los múltiples *warnings* de deprecación (`IPluginV2...`), de variables sin usar y de política CMake CMP0104 que acompañan al build son benignos: solo las líneas con `error:` detienen la compilación. |
| `nvcc fatal: Unsupported gpu architecture 'sm_50'` o `Code generation for 'FP16' ... compute capability less than '5.3'` | GPU Coder compila para su arquitectura por defecto (sm_50), eliminada del nvcc de CUDA 13. El script de despliegue ya declara la GPU real de la Orin con `cfgGen.GpuConfig.ComputeCapability = "8.7"`; verificar que esa línea está presente. |
| `Number of image channels of input passed to the predict method must be a code generation time constant` | La imagen de `rosReadImage` es de tamaño variable y el generador de código no conoce sus canales. El nodo ya lo resuelve fijándolos con indexación de rango constante (`I = I(:, :, 1:3);` justo después de `rosReadImage`); verificar que esa línea está presente si se modificó el entry-point. |
| `E: Malformed entry 1 in list file /etc/apt/sources.list.d/ros2.list` | La línea del repositorio se corrompió al copiarla desde un visor de Markdown (aparece texto como `%InlineEquation`). Borrar con `sudo rm /etc/apt/sources.list.d/ros2.list` y repetir el `echo` del paso 5, verificando con `cat`. |
| `nvcc: command not found` | Componentes del SDK no instalados → paso 2 (`sudo apt install nvidia-jetpack`). Si `/usr/local/cuda-XX.X` existe pero `/usr/local/cuda` no, crear el enlace simbólico (paso 2). |
| `ssh usuario@IP 'nvcc --version'` falla desde el host pero `nvcc` funciona en la Jetson | Las variables no están al **inicio** de `~/.bashrc` → repetir paso 4 verificando con `head -5 ~/.bashrc`. |
| `codegen` falla en la compilación remota | Variables CUDA no visibles por SSH no interactivo (paso 4), o incompatibilidad CUDA 13/TensorRT 10.16 con GPU Coder → probar `inferenceLibrary = "cudnn"` en el script de despliegue (ver recuadro del paso 1). |
| `ros2 topic list` en MATLAB no muestra `/image_raw` | `ROS_DOMAIN_ID` distinto, equipos en subredes diferentes o firewall bloqueando UDP (DDS usa multicast para el descubrimiento). |
| `/image_raw` existe pero sin datos (`hz` no responde) | El driver no accede a la cámara: revisar `v4l2-ctl --list-devices` y el parámetro `video_device`. |
| El nodo desplegado corre pero no publica resultados | El driver de cámara no está activo, o publica con otro `encoding` (debe ser `rgb8`). |
| FPS de video muy bajos en el host | Usar Ethernet en lugar de WiFi, o reducir `image_size` del driver. |
| Falta de memoria al compilar o inferir | Cerrar el escritorio gráfico: `sudo systemctl isolate multi-user.target` (volver con `sudo systemctl isolate graphical.target`). |
| El primer arranque del nodo tarda minutos | Normal: TensorRT construye y optimiza el motor de inferencia FP16 en el primer inicio. |
