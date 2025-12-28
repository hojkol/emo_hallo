import os
import platform
import sys
from uuid import uuid4

import streamlit as st
from loguru import logger

# Add the root directory of the project to the system path to allow importing modules from the project
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    print("******** sys.path ********")
    print(sys.path)
    print("")

from app.config import config
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import llm, voice
from app.services import task as tm
from app.utils import utils

st.set_page_config(
    page_title="Emo_hallo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/hojkol/Emo_hallo/issues",
        "About": "# Emo_hallo\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/hojkol/Emo_hallo",
    },
)


streamlit_style = """
<style>
h1 {
    padding-top: 0 !important;
}

/* 优化按钮样式 */
div[data-testid="column"] button {
    transition: all 0.3s ease;
}

div[data-testid="column"] button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* 图片容器样式 - 添加悬停效果 */
div[data-testid="stImage"] {
    border-radius: 10px;
    overflow: hidden;
    transition: all 0.3s ease;
    border: 3px solid #e0e0e0;
}

div[data-testid="stImage"]:hover {
    border-color: #1E88E5 !important;
    box-shadow: 0 0 20px rgba(30, 136, 229, 0.4);
    transform: scale(1.01);
}

/* 音频播放器样式优化 */
audio {
    width: 100%;
    border-radius: 5px;
}

/* 容器边框优化 */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    border-radius: 15px;
}

/* 信息框样式 */
div[data-baseweb="notification"] {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 文件上传器样式 */
div[data-testid="stFileUploader"] {
    border-radius: 10px;
    transition: all 0.3s ease;
}

div[data-testid="stFileUploader"]:hover {
    background-color: rgba(30, 136, 229, 0.05);
}

/* 删除按钮特殊样式 */
button[kind="secondary"]:has-text("❌") {
    background-color: #ff4444;
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    font-size: 20px;
    transition: all 0.3s ease;
}

button[kind="secondary"]:has-text("❌"):hover {
    background-color: #cc0000;
    transform: rotate(90deg) scale(1.1);
}

/* 全屏图片查看样式 */
.fullsize-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.95);
    z-index: 9999;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* 进度条样式 */
div[data-testid="stProgress"] > div {
    border-radius: 10px;
}

/* 隐藏文件上传器的历史记录列表 */
div[data-testid="stFileUploader"] ul {
    display: none !important;
}

div[data-testid="stFileUploader"] li {
    display: none !important;
}
</style>
"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# 定义资源目录
font_dir = os.path.join(root_dir, "resource", "fonts")
song_dir = os.path.join(root_dir, "resource", "songs")
i18n_dir = os.path.join(root_dir, "emo_hallo", "i18n")
config_file = os.path.join(root_dir, "emo_hallo", ".streamlit", "emo_hallo.toml")
system_locale = utils.get_system_locale()


if "video_subject" not in st.session_state:
    st.session_state["video_subject"] = ""
if "video_script" not in st.session_state:
    st.session_state["video_script"] = ""
if "video_terms" not in st.session_state:
    st.session_state["video_terms"] = ""
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = config.ui.get("language", system_locale)
if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = {}  # {id: file_object}
if "uploaded_audios" not in st.session_state:
    st.session_state["uploaded_audios"] = {}  # {id: file_object}
if "selected_image_id" not in st.session_state:
    st.session_state["selected_image_id"] = None
if "selected_audio_id" not in st.session_state:
    st.session_state["selected_audio_id"] = None
if "uploaded_image_names" not in st.session_state:
    st.session_state["uploaded_image_names"] = set()  # 已上传的图片名称集合
if "uploaded_audio_names" not in st.session_state:
    st.session_state["uploaded_audio_names"] = set()  # 已上传的音频名称集合

# 加载语言文件
locales = utils.load_locales(i18n_dir)

# 创建一个顶部栏，包含标题和语言选择
title_col, lang_col = st.columns([3, 1])

with title_col:
    st.title(f"Talking Head Generation v{config.project_version}")

with lang_col:
    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get("ui_language", ""):
            selected_index = i

    selected_language = st.selectbox(
        "Language / 语言",
        options=display_languages,
        index=selected_index,
        key="top_language_selector",
        label_visibility="collapsed",
    )
    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state["ui_language"] = code
        config.ui["language"] = code

support_locales = [
    "zh-CN",
    "zh-HK",
    "en-US",
    "zh-TW",
    "de-DE",
    "fr-FR",
    "vi-VN",
    "th-TH",
]


def get_all_fonts():
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".ttc"):
                fonts.append(file)
    fonts.sort()
    return fonts


def get_all_songs():
    songs = []
    for root, dirs, files in os.walk(song_dir):
        for file in files:
            if file.endswith(".mp3"):
                songs.append(file)
    return songs


def open_task_folder(task_id):
    try:
        sys = platform.system()
        path = os.path.join(root_dir, "storage", "tasks", task_id)
        if os.path.exists(path):
            if sys == "Windows":
                os.system(f"start {path}")
            if sys == "Darwin":
                os.system(f"open {path}")
    except Exception as e:
        logger.error(e)


def scroll_to_bottom():
    js = """
    <script>
        console.log("scroll_to_bottom");
        function scroll(dummy_var_to_force_repeat_execution){
            var sections = parent.document.querySelectorAll('section.main');
            console.log(sections);
            for(let index = 0; index<sections.length; index++) {
                sections[index].scrollTop = sections[index].scrollHeight;
            }
        }
        scroll(1);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)


def init_log():
    logger.remove()
    _lvl = "DEBUG"

    def format_record(record):
        # 获取日志记录中的文件全路径
        file_path = record["file"].path
        # 将绝对路径转换为相对于项目根目录的路径
        relative_path = os.path.relpath(file_path, root_dir)
        # 更新记录中的文件路径
        record["file"].path = f"./{relative_path}"
        # 返回修改后的格式字符串
        # 您可以根据需要调整这里的格式
        record["message"] = record["message"].replace(root_dir, ".")

        _format = (
            "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
            + "<level>{level}</> | "
            + '"{file.path}:{line}":<blue> {function}</> '
            + "- <level>{message}</>"
            + "\n"
        )
        return _format

    logger.add(
        sys.stdout,
        level=_lvl,
        format=format_record,
        colorize=True,
    )


init_log()

locales = utils.load_locales(i18n_dir)


def tr(key):
    loc = locales.get(st.session_state["ui_language"], {})
    return loc.get("Translation", {}).get(key, key)


# 创建基础设置折叠框
if not config.app.get("hide_config", False):
    with st.expander(tr("Basic Settings"), expanded=False):
        config_panels = st.columns(3)
        left_config_panel = config_panels[0]
        # middle_config_panel = config_panels[1]
        # right_config_panel = config_panels[2]
        middle_config_panel = st.container()

        # 左侧面板 - 日志设置
        with left_config_panel:
            # # 是否隐藏配置面板
            # hide_config = st.checkbox(
            #     tr("Hide Basic Settings"), value=config.app.get("hide_config", False)
            # )
            # config.app["hide_config"] = hide_config

            # 是否禁用日志显示
            hide_log = st.checkbox(
                tr("Hide Log"), value=config.ui.get("hide_log", False)
            )
            config.ui["hide_log"] = hide_log

        # 中间面板 - LLM 设置

        with middle_config_panel:
            st.markdown(f"###### {tr('Model Settings')}")

            # 第一行：运行本地模型 + 模型名称（始终显示在同一行）
            using_local_op = ["Yes", "No"]
            saved_using_index = 0
            saved_using_local = config.app.get("using_local", "Yes")
            for i, provider in enumerate(using_local_op):
                if provider.lower() == saved_using_local:
                    saved_using_index = i
                    break

            first_row_cols = st.columns(3)
            with first_row_cols[0]:
                using_local = st.selectbox(
                    tr("Running the Local Model"),
                    options=using_local_op,
                    index=saved_using_index,
                )
            config.app["using_local"] = using_local

            # 当前 LLM 提供商及模型名称（模型名称始终可编辑）
            current_llm_provider = config.app.get("llm_provider", "OpenAI").lower()
            current_llm_model_name = config.app.get(
                f"{current_llm_provider}_model_name", ""
            )
            # 当运行本地模型时，强制使用 EmoHallo 作为模型名称
            if using_local == "Yes":
                current_llm_model_name = "EmoHallo"
            with first_row_cols[1]:
                st_llm_model_name = st.text_input(
                    tr("Model Name"),
                    value=current_llm_model_name,
                    key=f"{current_llm_provider}_model_name_input",
                )
            if st_llm_model_name:
                config.app[f"{current_llm_provider}_model_name"] = st_llm_model_name

            # 其余配置块：仅在不使用本地模型时显示，且从新的一行开始
            grid_state = {"index": 0, "cols": None}

            def render_in_grid(render_callable):
                if grid_state["index"] % 3 == 0:
                    grid_state["cols"] = st.columns(3)
                col = grid_state["cols"][grid_state["index"] % 3]
                grid_state["index"] += 1
                with col:
                    return render_callable()

            if using_local == "No":
                llm_providers = [
                    "OpenAI",
                    "Moonshot",
                    "Azure",
                    "Qwen",
                    "DeepSeek",
                    "Gemini",
                    "Ollama",
                    "G4f",
                    "OneAPI",
                    "Cloudflare",
                    "ERNIE",
                    "Pollinations",
                ]
                saved_llm_provider = config.app.get("llm_provider", "OpenAI").lower()
                saved_llm_provider_index = 0
                for i, provider in enumerate(llm_providers):
                    if provider.lower() == saved_llm_provider:
                        saved_llm_provider_index = i
                        break
                llm_provider = render_in_grid(
                    lambda: st.selectbox(
                        tr("llm_provider"),
                        options=llm_providers,
                        index=saved_llm_provider_index,
                    )
                )
                llm_helper = st.container()
                llm_provider = llm_provider.lower()
                config.app["llm_provider"] = llm_provider

                llm_api_key = config.app.get(f"{llm_provider}_api_key", "")
                llm_secret_key = config.app.get(
                    f"{llm_provider}_secret_key", ""
                )  # only for baidu ernie
                llm_base_url = config.app.get(f"{llm_provider}_base_url", "")
                llm_model_name = config.app.get(f"{llm_provider}_model_name", "")
                llm_account_id = config.app.get(f"{llm_provider}_account_id", "")

                tips = ""
                if llm_provider == "ollama":
                    if not llm_model_name:
                        llm_model_name = "qwen:7b"
                    if not llm_base_url:
                        llm_base_url = "http://localhost:11434/v1"

                    with llm_helper:
                        tips = """
                                ##### Ollama配置说明
                                - **API Key**: 随便填写，比如 123
                                - **Base Url**: 一般为 http://localhost:11434/v1
                                    - 如果 `MoneyPrinterTurbo` 和 `Ollama` **不在同一台机器上**，需要填写 `Ollama` 机器的IP地址
                                    - 如果 `MoneyPrinterTurbo` 是 `Docker` 部署，建议填写 `http://host.docker.internal:11434/v1`
                                - **Model Name**: 使用 `ollama list` 查看，比如 `qwen:7b`
                                """

                if llm_provider == "openai":
                    if not llm_model_name:
                        llm_model_name = "gpt-3.5-turbo"
                    with llm_helper:
                        tips = """
                                ##### OpenAI 配置说明
                                > 需要VPN开启全局流量模式
                                - **API Key**: [点击到官网申请](https://platform.openai.com/api-keys)
                                - **Base Url**: 可以留空
                                - **Model Name**: 填写**有权限**的模型，[点击查看模型列表](https://platform.openai.com/settings/organization/limits)
                                """

                if llm_provider == "moonshot":
                    if not llm_model_name:
                        llm_model_name = "moonshot-v1-8k"
                    with llm_helper:
                        tips = """
                                ##### Moonshot 配置说明
                                - **API Key**: [点击到官网申请](https://platform.moonshot.cn/console/api-keys)
                                - **Base Url**: 固定为 https://api.moonshot.cn/v1
                                - **Model Name**: 比如 moonshot-v1-8k，[点击查看模型列表](https://platform.moonshot.cn/docs/intro#%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8)
                                """
                if llm_provider == "oneapi":
                    if not llm_model_name:
                        llm_model_name = (
                            "claude-3-5-sonnet-20240620"  # 默认模型，可以根据需要调整
                        )
                    with llm_helper:
                        tips = """
                            ##### OneAPI 配置说明
                            - **API Key**: 填写您的 OneAPI 密钥
                            - **Base Url**: 填写 OneAPI 的基础 URL
                            - **Model Name**: 填写您要使用的模型名称，例如 claude-3-5-sonnet-20240620
                            """

                if llm_provider == "qwen":
                    if not llm_model_name:
                        llm_model_name = "qwen-max"
                    with llm_helper:
                        tips = """
                                ##### 通义千问Qwen 配置说明
                                - **API Key**: [点击到官网申请](https://dashscope.console.aliyun.com/apiKey)
                                - **Base Url**: 留空
                                - **Model Name**: 比如 qwen-max，[点击查看模型列表](https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction#3ef6d0bcf91wy)
                                """

                if llm_provider == "g4f":
                    if not llm_model_name:
                        llm_model_name = "gpt-3.5-turbo"
                    with llm_helper:
                        tips = """
                                ##### gpt4free 配置说明
                                > [GitHub开源项目](https://github.com/xtekky/gpt4free)，可以免费使用GPT模型，但是**稳定性较差**
                                - **API Key**: 随便填写，比如 123
                                - **Base Url**: 留空
                                - **Model Name**: 比如 gpt-3.5-turbo，[点击查看模型列表](https://github.com/xtekky/gpt4free/blob/main/g4f/models.py#L308)
                                """
                if llm_provider == "azure":
                    with llm_helper:
                        tips = """
                                ##### Azure 配置说明
                                > [点击查看如何部署模型](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/create-resource)
                                - **API Key**: [点击到Azure后台创建](https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI)
                                - **Base Url**: 留空
                                - **Model Name**: 填写你实际的部署名
                                """

                if llm_provider == "gemini":
                    if not llm_model_name:
                        llm_model_name = "gemini-1.0-pro"

                    with llm_helper:
                        tips = """
                                ##### Gemini 配置说明
                                > 需要VPN开启全局流量模式
                                - **API Key**: [点击到官网申请](https://ai.google.dev/)
                                - **Base Url**: 留空
                                - **Model Name**: 比如 gemini-1.0-pro
                                """

                if llm_provider == "deepseek":
                    if not llm_model_name:
                        llm_model_name = "deepseek-chat"
                    if not llm_base_url:
                        llm_base_url = "https://api.deepseek.com"
                    with llm_helper:
                        tips = """
                                ##### DeepSeek 配置说明
                                - **API Key**: [点击到官网申请](https://platform.deepseek.com/api_keys)
                                - **Base Url**: 固定为 https://api.deepseek.com
                                - **Model Name**: 固定为 deepseek-chat
                                """

                if llm_provider == "ernie":
                    with llm_helper:
                        tips = """
                                ##### 百度文心一言 配置说明
                                - **API Key**: [点击到官网申请](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)
                                - **Secret Key**: [点击到官网申请](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)
                                - **Base Url**: 填写 **请求地址** [点击查看文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E)
                                """

                if llm_provider == "pollinations":
                    if not llm_model_name:
                        llm_model_name = "default"
                    with llm_helper:
                        tips = """
                                ##### Pollinations AI Configuration
                                - **API Key**: Optional - Leave empty for public access
                                - **Base Url**: Default is https://text.pollinations.ai/openai
                                - **Model Name**: Use 'openai-fast' or specify a model name
                                """

                if tips and config.ui["language"] == "zh":
                    st.warning(
                        "中国用户建议使用 **DeepSeek** 或 **Moonshot** 作为大模型提供商\n- 国内可直接访问，不需要VPN \n- 注册就送额度，基本够用"
                    )
                    st.info(tips)

                st_llm_api_key = render_in_grid(
                    lambda: st.text_input(
                        tr("API Key"), value=llm_api_key, type="password"
                    )
                )
                st_llm_base_url = render_in_grid(
                    lambda: st.text_input(tr("Base Url"), value=llm_base_url)
                )

                if st_llm_api_key:
                    config.app[f"{llm_provider}_api_key"] = st_llm_api_key
                if st_llm_base_url:
                    config.app[f"{llm_provider}_base_url"] = st_llm_base_url

                if llm_provider == "ernie":
                    st_llm_secret_key = render_in_grid(
                        lambda: st.text_input(
                            tr("Secret Key"), value=llm_secret_key, type="password"
                        )
                    )
                    config.app[f"{llm_provider}_secret_key"] = st_llm_secret_key

                if llm_provider == "cloudflare":
                    st_llm_account_id = render_in_grid(
                        lambda: st.text_input(
                            tr("Account ID"), value=llm_account_id
                        )
                    )
                    if st_llm_account_id:
                        config.app[f"{llm_provider}_account_id"] = st_llm_account_id



llm_provider = config.app.get("llm_provider", "").lower()

params = VideoParams(video_subject="")
uploaded_files = []

# 创建居中布局 - 上传区域占1/3宽度
col_left, col_image, clo_audio, col_right = st.columns([1, 2, 2, 1])

with col_image:
    # 图片上传区域
    with st.container(border=True):
        st.markdown("### 📷 " + tr("Upload Image"))

        # 图片预览区 - 每行最多两个容器
        st.markdown("#### " + tr("Image Preview"))
        if st.session_state["uploaded_images"]:
            image_ids = list(st.session_state["uploaded_images"].keys())
            # 每行两个，分行显示
            for i in range(0, len(image_ids), 2):
                # 显示两个图片
                img_display_cols = st.columns([1, 1])

                with img_display_cols[0]:
                    img_id = image_ids[i]
                    img_file = st.session_state["uploaded_images"][img_id]
                    st.image(img_file, use_container_width=True)

                if i + 1 < len(image_ids):
                    with img_display_cols[1]:
                        img_id = image_ids[i + 1]
                        img_file = st.session_state["uploaded_images"][img_id]
                        st.image(img_file, use_container_width=True)

                # 显示文件名和操作按钮（在同一行）
                img_action_cols = st.columns([1.5, 0.25, 0.25, 1.5, 0.25, 0.25])

                # 第一个图片的文件名和按钮
                img_id = image_ids[i]
                img_file = st.session_state["uploaded_images"][img_id]

                with img_action_cols[0]:
                    st.markdown(f"**{img_file.name}**")

                with img_action_cols[1]:
                    if st.button("✅", key=f"use_img_{img_id}", use_container_width=True, help=tr("Use")):
                        st.session_state["selected_image_id"] = img_id
                        st.rerun()

                with img_action_cols[2]:
                    if st.button("❌", key=f"delete_img_{img_id}", use_container_width=True, help=tr("Delete")):
                        img_name = st.session_state["uploaded_images"][img_id].name
                        st.session_state["uploaded_image_names"].discard(img_name)
                        del st.session_state["uploaded_images"][img_id]
                        if st.session_state["selected_image_id"] == img_id:
                            st.session_state["selected_image_id"] = None
                        st.rerun()

                # 第二个图片的文件名和按钮
                if i + 1 < len(image_ids):
                    img_id = image_ids[i + 1]
                    img_file = st.session_state["uploaded_images"][img_id]

                    with img_action_cols[3]:
                        st.markdown(f"**{img_file.name}**")

                    with img_action_cols[4]:
                        if st.button("✅", key=f"use_img_{img_id}", use_container_width=True, help=tr("Use")):
                            st.session_state["selected_image_id"] = img_id
                            st.rerun()

                    with img_action_cols[5]:
                        if st.button("❌", key=f"delete_img_{img_id}", use_container_width=True, help=tr("Delete")):
                            img_name = st.session_state["uploaded_images"][img_id].name
                            st.session_state["uploaded_image_names"].discard(img_name)
                            del st.session_state["uploaded_images"][img_id]
                            if st.session_state["selected_image_id"] == img_id:
                                st.session_state["selected_image_id"] = None
                            st.rerun()
        else:
            st.info(tr("No images uploaded yet"))

        st.markdown("---")

        # 图片上传区域 - 不显示已上传文件
        # st.markdown("#### " + tr("Upload Image"))
        # st.caption("📝 " + tr("Supported formats") + ": JPG, JPEG, PNG, BMP")
        temp_images = st.file_uploader(
            tr("Choose images"),
            type=[".jpg", ".jpeg", ".png", ".bmp"],
            accept_multiple_files=True,
            key="image_uploader",
            label_visibility="collapsed"
        )

        # 添加新上传的图片到session state（只添加新文件，避免重复）
        if temp_images:
            files_added = False
            for temp_image in temp_images:
                # 验证文件类型
                img_ext = temp_image.name.split('.')[-1].lower()
                if img_ext in ["jpg", "jpeg", "png", "bmp"]:
                    # 只有当这个文件名还没有被上传过时才添加
                    if temp_image.name not in st.session_state["uploaded_image_names"]:
                        img_id = str(uuid4())
                        st.session_state["uploaded_images"][img_id] = temp_image
                        st.session_state["uploaded_image_names"].add(temp_image.name)
                        files_added = True
                else:
                    st.warning(f"⚠️ " + tr("File type not supported") + f": {temp_image.name}")

            # 刷新页面显示预览
            if files_added:
                st.rerun()

with clo_audio:
    # 音频上传区域
    with st.container(border=True):
        st.markdown("### 🎵 " + tr("Upload Audio"))

        # 显示所有音频文件列表 - 最上方
        st.markdown("#### " + tr("Audio Files"))
        if st.session_state["uploaded_audios"]:
            audio_ids = list(st.session_state["uploaded_audios"].keys())
            for audio_id in audio_ids:
                audio_file = st.session_state["uploaded_audios"][audio_id]

                # 文件名、播放器和操作按钮在同一行
                # 列布局：[文件名, 播放器, Use按钮, Delete按钮]
                audio_cols = st.columns([2, 3.5, 0.5, 0.5])

                # 文件名
                with audio_cols[0]:
                    st.markdown(f"🎵 **{audio_file.name}**")

                # 音频播放器
                with audio_cols[1]:
                    st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")

                # Use 按钮
                with audio_cols[2]:
                    if st.button("✅", key=f"use_audio_{audio_id}", use_container_width=True, help=tr("Use")):
                        st.session_state["selected_audio_id"] = audio_id
                        st.rerun()

                # Delete 按钮
                with audio_cols[3]:
                    if st.button("❌", key=f"delete_audio_{audio_id}", use_container_width=True, help=tr("Delete")):
                        # 删除文件名记录，防止重复上传检测出错
                        audio_name = st.session_state["uploaded_audios"][audio_id].name
                        st.session_state["uploaded_audio_names"].discard(audio_name)
                        del st.session_state["uploaded_audios"][audio_id]
                        if st.session_state["selected_audio_id"] == audio_id:
                            st.session_state["selected_audio_id"] = None
                        st.rerun()

                st.markdown("")  # 空行分隔
        else:
            st.info(tr("No audio files yet"))

        st.markdown("---")

        # 音频上传区域 - 不显示已上传文件
        # st.markdown("#### " + tr("Upload Audio"))
        # st.caption("📝 " + tr("Supported formats") + ": MP3, WAV, OGG, M4A")
        temp_audios = st.file_uploader(
            tr("Choose audio files"),
            type=[".mp3", ".wav", ".ogg", ".m4a"],
            accept_multiple_files=True,
            key="audio_uploader",
            label_visibility="collapsed"
        )

        # 添加新上传的音频到session state（只添加新文件，避免重复）
        if temp_audios:
            files_added = False
            for temp_audio in temp_audios:
                # 验证文件类型
                audio_ext = temp_audio.name.split('.')[-1].lower()
                if audio_ext in ["mp3", "wav", "ogg", "m4a"]:
                    # 只有当这个文件名还没有被上传过时才添加
                    if temp_audio.name not in st.session_state["uploaded_audio_names"]:
                        audio_id = str(uuid4())
                        st.session_state["uploaded_audios"][audio_id] = temp_audio
                        st.session_state["uploaded_audio_names"].add(temp_audio.name)
                        files_added = True
                else:
                    st.warning(f"⚠️ " + tr("File type not supported") + f": {temp_audio.name}")

            # 刷新页面显示预览
            if files_added:
                st.rerun()

# Generate按钮和其他区域
col_generate, = st.columns([1])

with col_generate:
    # 选择图片和音频区域 - 同一行显示
    # 并排显示 Select Image 和 Select Audio（2列布局）
    select_cols = st.columns(2)

    # Select Image
    with select_cols[0]:
        if st.session_state["uploaded_images"]:
            image_options = {img_id: st.session_state["uploaded_images"][img_id].name
                           for img_id in st.session_state["uploaded_images"]}
            # 获取当前选中的图片名称，如果没有则显示第一张
            if st.session_state["selected_image_id"] is None and image_options:
                current_img_name = list(image_options.values())[0]
            elif st.session_state["selected_image_id"] in st.session_state["uploaded_images"]:
                current_img_name = st.session_state["uploaded_images"][st.session_state["selected_image_id"]].name
            else:
                current_img_name = None

            selected_img_name = st.selectbox(
                "📷 " + tr("Choose an image"),
                options=list(image_options.values()),
                index=list(image_options.values()).index(current_img_name) if current_img_name in image_options.values() else 0,
                key="select_image_gen",
            )
            # 更新选中的image_id
            for img_id, name in image_options.items():
                if name == selected_img_name:
                    st.session_state["selected_image_id"] = img_id
                    break
        else:
            st.warning(tr("Please upload an image first"))

    # Select Audio
    with select_cols[1]:
        if st.session_state["uploaded_audios"]:
            audio_options = {audio_id: st.session_state["uploaded_audios"][audio_id].name
                           for audio_id in st.session_state["uploaded_audios"]}
            # 获取当前选中的音频名称，如果没有则显示第一个
            if st.session_state["selected_audio_id"] is None and audio_options:
                current_audio_name = list(audio_options.values())[0]
            elif st.session_state["selected_audio_id"] in st.session_state["uploaded_audios"]:
                current_audio_name = st.session_state["uploaded_audios"][st.session_state["selected_audio_id"]].name
            else:
                current_audio_name = None

            selected_audio_name = st.selectbox(
                "🎵 " + tr("Choose an audio"),
                options=list(audio_options.values()),
                index=list(audio_options.values()).index(current_audio_name) if current_audio_name in audio_options.values() else 0,
                key="select_audio_gen",
            )
            # 更新选中的audio_id
            for audio_id, name in audio_options.items():
                if name == selected_audio_name:
                    st.session_state["selected_audio_id"] = audio_id
                    break
        else:
            st.warning(tr("Please upload an audio file first"))

    # 生成按钮 - 单独在下一行
    col_btn_spacer, col_btn = st.columns([2, 1])
    with col_btn:
        generate_btn = st.button(
            "🚀 " + tr("Generate"),
            type="primary",
            use_container_width=True,
            key="generate_talking_head"
        )

    st.markdown("---")

    # 生成按钮点击后的处理逻辑
    if generate_btn:
        if not st.session_state["uploaded_images"]:
            st.error("❌ " + tr("Please upload an image first"))
        elif not st.session_state["uploaded_audios"]:
            st.error("❌ " + tr("Please upload an audio file first"))
        elif st.session_state["selected_image_id"] is None:
            st.error("❌ " + tr("Please select an image"))
        elif st.session_state["selected_audio_id"] is None:
            st.error("❌ " + tr("Please select an audio file"))
        else:
            # TODO: 将选中的图片和音频传给后台模型处理
            selected_image = st.session_state["uploaded_images"][st.session_state["selected_image_id"]]
            selected_audio = st.session_state["uploaded_audios"][st.session_state["selected_audio_id"]]
            st.success(f"✅ " + tr("Generating with image") + f": {selected_image.name}, " + tr("audio") + f": {selected_audio.name}")

    # 进度条容器
    with st.container(border=True):
        st.markdown("### ⏳ " + tr("Task Progress"))
        progress_bar = st.progress(0)
        progress_text = st.empty()
        # 后端功能待实现
        progress_text.text("🟢 " + tr("Ready"))

    # 历史记录容器
    with st.container(border=True):
        st.markdown("### 🎬 " + tr("Recent Creations"))

        # 示例:显示历史视频列表(后端功能待实现)
        # 这里可以从数据库或文件系统读取历史生成的视频
        recent_videos = []  # TODO: 从后端获取历史视频列表

        if recent_videos:
            # 使用列显示视频缩略图
            cols = st.columns(3)
            for i, video_path in enumerate(recent_videos):
                with cols[i % 3]:
                    st.video(video_path)
        else:
            st.info("📂 " + tr("No recent creations yet"))


config.save_config()
