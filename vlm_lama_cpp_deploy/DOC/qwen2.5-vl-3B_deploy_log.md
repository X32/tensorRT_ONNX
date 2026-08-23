 编译llama.cpp
 构建 llama.cpp（启用 CUDA，吃 Orin 的 GPU）
```bash
sudo apt update
sudo apt install -y build-essential cmake git curl

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# 启用 CUDA backend（JetPack 6.2 自带 CUDA 12.6，Orin = sm_87 会自动识别）
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```


 下载模型
 huggingface-cli download Taoufik/Qwen2.5-VL-3B-Instruct-Q4_K_M-GGUF  --local-dir ./qwen_vl_gguf/
 

测试模型的文字能力

命令行测试

./build/bin/llama-cli \
  -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
  -p "解释量子计算的基本原理" \
  -n 512 \
  --ctx-size 2048

> 解释量子计算的基本原理
量子计算是利用量子力学原理来进行计算的一种新型计算方式。与经典计算机使用二进制的0和1不同，量子计算使用了量子比特（qubit），它可以同时表示0和1的状态。

量子计算的基本原理是利用量子叠加和量子纠缠。量子叠加是指一个量子比特可以同时处于0和1的状态，这意味着一个量子计算可以同时处理大量的数据。量子纠缠则是指两个量子比特之间存在一种特殊的关系，它们的状态是相互关联的，一个量子比特的状态变化会立刻反映到另一个量子比特上，这种现象称为“量子纠缠”。

量子计算的优点在于它可以在短时间内解决某些问题，而经典计算机需要很长时间才能完成。例如，量子计算可以在短时间内解决经典的NP难问题，如大数分解问题，而经典计算机可能需要数十年才能完成。

但是，量子计算也面临一些挑战，如量子计算的错误率、量子纠缠的保持、量子比特的稳定等。目前，量子计算的研究还在初级阶段，需要大量的研究和改进，才能让量子计算成为主流的计算方式。

[ Prompt: 147.6 t/s | Generation: 23.1 t/s ]

启动server测试文本处理能力

./build/bin/llama-server \
    -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 2048

curl http://localhost:8080/v1/chat/completions\
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-3.5-turbo",
      "messages": [
        {"role": "user", "content":  "你好，请用一句话介绍一下你自己"}
      ],
      "max_tokens": 128
    }'
{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"您好，我是来自阿里云的大规模语言模型，我叫通义千问。"}}],"created":1787474018,"model":"/home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf","system_fingerprint":"b9972-c92e806d1","object":"chat.completion","usage":{"completion_tokens":19,"prompt_tokens":25,"total_tokens":44,"prompt_tokens_details":{"cached_tokens":0}},"id":"chatcmpl-gR0ECoOFEQ7vgBnlNBvAiVnItcf2enbe","timings":{"cache_n":0,"prompt_n":25,"prompt_ms":158.758,"prompt_per_token_ms":6.35032,"prompt_per_second":157.47237934466293,"predicted_n":19,"predicted_ms":981.082,"predicted_per_token_ms":51.635894736842104,"predicted_per_second":19.36637304527043}}(base)

下载图形处理头

wget https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf



启动模型

CLI模式
./build/bin/llama-cli \
  -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
--mmproj /home/zy/Desktop/workplace/models/qwen_vl_gguf/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
  --image /home/zy/Desktop/workplace/tensorRT/vlm_lama_cpp/test_resized.jpg \
  -p "这张图片里有什么？" \
  -n 256 \
  --ctx-size 2048



Loaded media from '/home/zy/Desktop/workplace/tensorRT/vlm_lama_cpp/test_resized.jpg'

> 这张图片里有什么？
这张图片展示了一个冰箱的内部，里面装满了各种饮料。具体来说，冰箱门的上部和下部分别装有不同种类的饮料，包括碳酸饮料、果汁和运动饮料等。冰箱门的左侧还有一张纸，纸上有几只小动物的剪影。冰箱内部的格局整齐，看起来非常干净和整洁。

[ Prompt: 120.1 t/s | Generation: 9.0 t/s ]

server模式

 ./build/bin/llama-server \
    -m /home/zy/Desktop/workplace/models/qwen_vl_gguf/qwen2.5-vl-3b-instruct-q4_k_m.gguf \
    --mmproj /home/zy/Desktop/workplace/models/qwen_vl_gguf/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
    -ngl 99 \
    --ctx-size 3072 \
    --host 127.0.0.1 \
    --port 8080


    测试模型的图片识别能力

    python test_vlm.py --url http://localhost:8080 --image ./test.jpg --max-tokens 512 --resize 768 

    python test_vlm.py --url http://localhost:8080 --image ./test.jpg --max-tokens 512 --resize 768
============================================================
🧪 Qwen2.5-VL llama.cpp 服务测试
============================================================
✅ 服务器在线: http://localhost:8080

[模式] 视觉理解
[提问] 描述这张图片的内容
------------------------------------------------------------
📐 已缩小到 432x768 (长边 768px)
🖼️  图片: test_resized.jpg (image/jpeg, 0.07 MB)

💬 回答:
这张图片展示了一个冰箱的多个视角，每个视角都有不同的饮料和食物。冰箱的门是打开的，可以看到内部的多个隔层。从上到下，冰箱门上依次排列着各种饮料，包括汽水、果汁、苏打水等。还有一些蔬菜和水果，如葡萄、苹果、柠檬和橙子。冰箱内部的隔层和抽屉里也摆放着各种饮料和食物，如矿泉水、果汁、酸奶和零食。整体感觉是一个装满各种饮料和食物的冰箱，看起来非常丰富和多汁。

------------------------------------------------------------
⏱️  总耗时: 6.99s
📊 生成 token: 117 | 速度: 16.7 tok/s
📊 完整 usage: {'completion_tokens': 117, 'prompt_tokens': 430, 'total_tokens': 547, 'prompt_tokens_details': {'cached_tokens': 0}}


 python test_ocr.py --image ./document.jpg --type document
============================================================
🔍 OCR 测试: ./document.jpg
📝 类型: document
============================================================
🖼️  图片: document.jpg (image/jpeg, 0.13 MB)

📄 识别结果:
标题：为什么要把看图和说话分存

正文：
1. **lana.cpp 故意把“看图”和“说话”分存：**
   - 文本部分可以大胆量化(q4_K_M)，体积压到1/4，节省内存；
   - 视觉部分保持f16——量化视觉编码器容易掉点（尤其grouning坐标精度），所以官方给的mmpoj都是f16全精度。

格式：
- 标题：为什么要把看图和说话分存
- 正文：1. **lana.cpp 故意把“看图”和“说话”分存：**
   - 文本部分可以大胆量化(q4_K_M)，体积压到1/4，节省内存；
   - 视觉部分保持f16——量化视觉编码器容易掉点（尤其grouning坐标精度），所以官方给的mmpoj都是f16全精度。
============================================================
