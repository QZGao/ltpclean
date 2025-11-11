from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import queue
from threading import Thread, Event
import torch
import time
from utils import get_data
from algorithm import Algorithm
from models.vae.sdvae import SDVAE
import config.configTrain as cfg
from infer_test import init_simulator, get_web_img
import os
"""Model info"""
frame_rate = 1 / 20

model = Algorithm(cfg.model_name, cfg.device)
state_dict = torch.load(os.path.join("ckpt",cfg.model_path),map_location=cfg.device,weights_only=False)
model.load_state_dict(state_dict["network_state_dict"],strict=False)
model.eval().to(cfg.device)
vae = SDVAE()
vae_state_dict = torch.load(cfg.vae_model, map_location=cfg.device,weights_only=False)
vae.load_state_dict(vae_state_dict['network_state_dict'], strict=False)
vae.eval().to(cfg.device)



init_data = get_data()

app = Flask(__name__)
socketio = SocketIO(app,
                    cors_allowed_origins="*",
                    logger=False,
                    engineio_logger=False,
                    transports=['websocket', 'polling'],
                    allow_upgrades=True)

"""User Info"""
user_zeta = {}
user_queues = {}
user_transfer_queues = {}  # 用于传递GPU tensor到传输线程
user_cmd = {}
user_config = {}
user_threads = {}
online_player = {}

default_cmd_str = "NULL"  # 与前端保持一致


def numpy2imgstr(image):
    image = get_web_img(image)
    img = Image.fromarray(image)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def get_jave_7action(key):
    if key == "r":
        action = 2
    elif key == "rj":
        action = 6
    elif key == "l":
        action = 1
    elif key == "lj":
        action = 5
    elif key == "j":
        action = 4
    elif key == "s":
        action = 8
    elif key == "NULL" or key == "None":  # 兼容两种写法
        action = 0
    else:
        action = 0
    return [action]


def get_user_config(data):
    ret = {
        'random_init': False,
        'block': False,
        'denosing_step': 4,
    }
    random_init = data['random_init']
    block = data['block']
    denosing_step = data['denosing_step']
    if random_init == 'true':
        ret['random_init'] = True
    if block == 'true':
        ret['block'] = True
    # 支持更多denoising step选项
    if denosing_step in ['1', '2', '4', '8']:
        ret['denosing_step'] = int(denosing_step)
    return ret


@socketio.on('key_press')
def handle_key_press(data):
    user_id = request.sid
    key = data['key']
    user_cmd[user_id] = key


@socketio.on('key_release')
def handle_key_release(data):
    user_id = request.sid
    key = data['key']
    if user_cmd[user_id] == 'lj':
        if key == 'l':
            user_cmd[user_id] = 'j'
        elif key == 'j':
            user_cmd[user_id] = 'l'
    elif user_cmd[user_id] == 'rj':
        if key == 'r':
            user_cmd[user_id] = 'j'
        elif key == 'j':
            user_cmd[user_id] = 'r'
    else:
        user_cmd[user_id] = default_cmd_str


@socketio.on('start_game')
def button_clicked(data):
    user_id = request.sid
    player_config = get_user_config(data)
    user_config[user_id] = player_config
    
    # 重置游戏状态：清空之前的命令和状态
    user_cmd[user_id] = default_cmd_str  # 重置按键命令
    
    # 清空队列中的旧数据（不删除队列本身，因为线程还在使用）
    while True:
        try:
            user_queues[user_id].get_nowait()
        except queue.Empty:
            break
    while True:
        try:
            user_transfer_queues[user_id].get_nowait()
        except queue.Empty:
            break
    
    # init simulator first frame
    with torch.no_grad():
        if player_config['random_init']:
            batch_data = get_data(if_random=True)
        else:
            # 创建副本避免修改全局 init_data
            batch_data = {"observations": init_data["observations"].clone()}        
        # 转换数据格式：从 [1, 1, 3, 256, 256] 转为 [1, 3, 256, 256]（与get_img_data格式一致）
        if batch_data["observations"].dim() == 5:
            batch_data["observations"] = batch_data["observations"].squeeze(1)  # [1, 1, 3, 256, 256] -> [1, 3, 256, 256]
        
        zeta,obs = init_simulator(model, vae, batch_data)
        
        # 压缩初始帧从256x256到128x128
        obs_gpu = torch.nn.functional.interpolate(
            obs,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        obs_np = obs_gpu.cpu().numpy()

    user_zeta[user_id] = zeta  # 重置zeta状态
    # 使用压缩后的128x128图像
    user_queues[user_id].put((obs_np, "None", "init (128x128)"))

    # record player
    online_player[user_id] = user_id
    socketio.emit('update_person', {'num': len(online_player)})


def model_inference(user_id, stop_event):
    while not stop_event.is_set():
        if user_id in user_cmd.keys() and user_id in user_zeta.keys():
            start_time = time.time()
            key = user_cmd[user_id]
            block = user_config[user_id]['block']
            sampling_timesteps = user_config[user_id]['denosing_step']
            if key == default_cmd_str and block:
                continue

            action = get_jave_7action(key)
            action = torch.tensor(action, device=cfg.device).long()
            zeta = user_zeta[user_id]
            with torch.no_grad():
                # Diffusion step
                step_start = time.time()
                zeta, obs = model.df_model.step(zeta, action.float(), sampling_timesteps)
                step_time = time.time() - step_start
                
                # VAE decode
                decode_start = time.time()
                obs = vae.decode(obs / 0.1355)
                decode_time = time.time() - decode_start
                
                # 在GPU上压缩图像从256x256到128x128
                resize_start = time.time()
                obs_gpu = torch.nn.functional.interpolate(
                    obs,
                    size=(128, 128),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
            user_zeta[user_id] = zeta
            resize_time = time.time() - resize_start
            # 计算推理耗时（不包括传输时间）
            inference_time = time.time() - start_time
            
            # 将GPU tensor放入传输队列，让传输线程异步处理
            duration_prefix = f"{inference_time:.3f}s (step:{step_time:.3f} decode:{decode_time:.3f} resize:{resize_time:.3f}"
            user_transfer_queues[user_id].put((obs_gpu, key, duration_prefix))
            
            # 计算剩余时间（只考虑推理时间，传输在后台进行）
            rest_time = frame_rate - inference_time
            if rest_time > 0:
                time.sleep(rest_time)
        else:
            time.sleep(0.01)


def transfer_gpu_to_cpu(user_id, stop_event):
    """异步传输线程：专门负责GPU->CPU转换，不阻塞推理线程"""
    while not stop_event.is_set():
        try:
            # 非阻塞获取待传输的tensor
            obs_gpu, cmd_str_, duration_prefix= user_transfer_queues[user_id].get(timeout=0.01)
            
            # 执行GPU->CPU传输（.cpu()会自动等待GPU上的resize操作完成）
            transfer_start = time.time()
            obs = obs_gpu.cpu().numpy()
            transfer_time = time.time() - transfer_start
            
            # 验证尺寸（调试用，只在第一次出错时打印）
            if obs.shape != (3, 128, 128):
                if not hasattr(transfer_gpu_to_cpu, '_warned_shape'):
                    print(f"⚠️ 警告: 数组尺寸不正确，期望(3,128,128)，实际{obs.shape}")
                    transfer_gpu_to_cpu._warned_shape = True
            
            # 格式化传输时间字符串
            transfer_time_str = f"{transfer_time:.6f}" if transfer_time < 0.001 else f"{transfer_time:.3f}"
            
            # 组装完整的duration信息（duration_prefix已包含resize时间）
            duration = f"{duration_prefix} transfer:{transfer_time_str})"
            
            # 将转换后的numpy数组放入结果队列
            user_queues[user_id].put((obs, cmd_str_, duration))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"传输线程错误: {e}")
            time.sleep(0.01)


def send_results(user_id, stop_event):
    while not stop_event.is_set():
        if not user_queues[user_id].empty():
            obs, cmd_str_, dur = user_queues[user_id].get()
            socketio.emit('update_frame', {
                'image': numpy2imgstr(obs),
            }, room=user_id)
            socketio.emit('update_cmd', {'cmd': cmd_str_, 'dur': dur}, room=user_id)
        else:
            time.sleep(0.01)


@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    user_cmd[user_id] = default_cmd_str
    user_queues[user_id] = queue.Queue()
    user_transfer_queues[user_id] = queue.Queue()  # 新增：传输队列
    join_room(user_id)

    stop_event = Event()
    
    # 推理线程：运行模型，生成GPU tensor
    inference_thread = Thread(target=model_inference, args=(user_id, stop_event))
    inference_thread.daemon = True
    inference_thread.start()

    # 传输线程：异步执行GPU->CPU转换
    transfer_thread = Thread(target=transfer_gpu_to_cpu, args=(user_id, stop_event))
    transfer_thread.daemon = True
    transfer_thread.start()

    # 发送线程：将结果发送到前端
    result_thread = Thread(target=send_results, args=(user_id, stop_event))
    result_thread.daemon = True
    result_thread.start()

    user_threads[user_id] = (inference_thread, transfer_thread, result_thread, stop_event)
    socketio.emit('update_person', {'num': len(online_player)})


@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    leave_room(user_id)
    user_cmd.pop(user_id, None)
    user_queues.pop(user_id, None)
    user_transfer_queues.pop(user_id, None)  # 清理传输队列
    user_config.pop(user_id, None)
    online_player.pop(user_id, None)
    if user_id in user_threads:
        inference_thread, transfer_thread, result_thread, stop_event = user_threads.pop(user_id)
        stop_event.set()
        inference_thread.join()
        transfer_thread.join()
        result_thread.join()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    return render_template('debug_test.html')


@app.route('/test')
def test():
    return "Flask应用正常工作！"


@socketio.on('test')
def handle_test(data):
    emit('test_response', {'message': '测试成功', 'received': data})


if __name__ == '__main__':
    import os

    # 检查是否在Colab环境中运行
    try:
        import google.colab

        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # 方法1：尝试使用ngrok（需要token）
        ngrok_success = False
        try:
            from pyngrok import ngrok

            # 检查是否有ngrok token
            ngrok_token = os.environ.get('NGROK_AUTHTOKEN')
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)

            # 启动ngrok隧道
            public_url = ngrok.connect(1235)
            print(f"应用已启动！")
            print(f"本地地址: http://localhost:1235")
            print(f"公网地址: {public_url}")
            print(f"请在浏览器中打开: {public_url}")
            ngrok_success = True

        except Exception as e:
            print("ngrok启动失败，使用Colab内置端口转发")
            print("请使用: google.colab.kernel.proxyPort(1235) 获取公网URL")

        # 在Colab中运行
        socketio.run(app, debug=False, host='0.0.0.0', port=1235, allow_unsafe_werkzeug=True)

    else:
        # 本地运行
        socketio.run(app, debug=True, host='0.0.0.0', port=1235)
