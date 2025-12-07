import gradio as gr
import torch
import requests
import io
import base64
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel

# --- CONFIGURAZIONE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
SD_API_URL = "http://127.0.0.1:7860"

# --- PRESET DI STILE ---
STYLES = {
    "Nessuno (Puro)": "",
    "Cinematic / Film": ", cinematic shot, 35mm photograph, film grain, vignette, color graded, dramatic lighting, detailed texture, masterpiece",
    "Anime / Manga": ", anime style, studio ghibli, makoto shinkai, cel shaded, vibrant colors, detailed line art, 4k, wallpaper",
    "Cyberpunk": ", cyberpunk, neon lights, futuristic city, chrome, high tech, blade runner vibes, synthwave, volumetric lighting",
    "Fotorealistico": ", hyperrealistic, 8k, highly detailed, sharp focus, f/1.8, bokeh, unreal engine 5 render, global illumination, raw photo",
    "Digital Art": ", digital painting, artstation, concept art, smooth, sharp focus, illustration, fantasy",
    "Dark Fantasy": ", dark fantasy, gothic, gloomy, foggy, grotesque, intricate details, eldritch, terrifying, low key",
    "3D Render": ", 3d render, octane render, unreal engine, ray tracing, v-ray, clean, geometric, abstract"
}

DEFAULT_NEGATIVE = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, text, error, jpeg artifacts"

# --- CACHE MODELLI ---
loaded_models = {}

def load_models():
    """Carica i modelli in memoria solo se non ci sono già"""
    if "BLIP" not in loaded_models:
        print("Caricamento BLIP...")
        p = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        m = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        loaded_models["BLIP"] = (p, m)
    
    if "GPT2" not in loaded_models:
        print("Caricamento GPT-2...")
        t = GPT2Tokenizer.from_pretrained("succinctly/text2image-prompt-generator")
        m = GPT2LMHeadModel.from_pretrained("succinctly/text2image-prompt-generator").to(device)
        loaded_models["GPT2"] = (t, m)
    
    return loaded_models["BLIP"], loaded_models["GPT2"]

# --- FUNZIONI DI LOGICA ---

def analyze_image_logic(image, style, use_ai):
    """Logica per Tab 1: Immagine -> Prompt"""
    if image is None: return "Nessuna immagine.", "", DEFAULT_NEGATIVE
    
    (blip_proc, blip_model), (gpt_tok, gpt_model) = load_models()
    
    # 1. Analisi (BLIP)
    inputs = blip_proc(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=75)
    base_desc = blip_proc.decode(out[0], skip_special_tokens=True)
    
    # 2. Arricchimento (GPT-2)
    generated_part = base_desc
    if use_ai:
        input_ids = gpt_tok(base_desc, return_tensors="pt").input_ids.to(device)
        output = gpt_model.generate(input_ids, max_length=120, do_sample=True, temperature=0.9, top_k=50)
        generated_part = gpt_tok.decode(output[0], skip_special_tokens=True)
        if base_desc in generated_part:
            generated_part = generated_part.replace(base_desc, "", 1).strip()
    
    # 3. Prompt Finale
    style_tags = STYLES.get(style, "")
    final_prompt = f"{base_desc}, {generated_part} {style_tags}".strip(", ")
    
    return base_desc, final_prompt, DEFAULT_NEGATIVE

def text_expansion_logic(user_idea, style, temp, max_len):
    """Logica per Tab 2: Testo -> Prompt"""
    if not user_idea: return "", DEFAULT_NEGATIVE
    
    _, (gpt_tok, gpt_model) = load_models()
    
    # Arricchimento con parametri personalizzati
    input_ids = gpt_tok(user_idea, return_tensors="pt").input_ids.to(device)
    
    output = gpt_model.generate(
        input_ids, 
        max_length=int(max_len),    # Lunghezza massima
        do_sample=True, 
        temperature=float(temp),    # Creatività
        top_k=50
    )
    
    generated_text = gpt_tok.decode(output[0], skip_special_tokens=True)
    
    # Aggiunta stile
    style_tags = STYLES.get(style, "")
    final_prompt = f"{generated_text} {style_tags}".strip()
    
    # Pulizia
    final_prompt = final_prompt.replace("  ", " ").replace(",,", ",")
    
    return final_prompt, DEFAULT_NEGATIVE

def send_to_sd_api(prompt, negative_prompt, steps, cfg, batch_size, seed):
    """Invia a Stable Diffusion (funziona per entrambi i tab)"""
    real_seed = seed if seed != -1 else -1

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg,
        "width": 512,
        "height": 512,
        "batch_size": batch_size,
        "seed": real_seed,
        "sampler_name": "Euler a"
    }

    try:
        response = requests.post(url=f"{SD_API_URL}/sdapi/v1/txt2img", json=payload)
        if response.status_code == 200:
            r = response.json()
            images = []
            for i in r['images']:
                image = Image.open(io.BytesIO(base64.b64decode(i)))
                images.append(image)
            return images, f"✅ Fatto: {batch_size} immagini generate."
        else:
            return [], f"Errore Server SD: {response.status_code}"
    except Exception as e:
        return [], f"Errore: {str(e)}"

# --- INTERFACCIA GRAFICA (GUI) ---

with gr.Blocks(title="Ultimate AI Studio") as app:
    gr.Markdown("# 🎨 Ultimate AI Studio")
    
    # --- TAB 1: DALL'IMMAGINE ---
    with gr.Tab("📸 Image Workbench"):
        with gr.Row():
            # Input Sinistra
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Immagine Riferimento", height=250)
                btn_analyze = gr.Button("🔍 1. Analizza", variant="secondary")
                
                gr.Markdown("---")
                style_sel_img = gr.Dropdown(list(STYLES.keys()), value="Fotorealistico", label="Stile")
                
                with gr.Accordion("Parametri SD", open=False):
                    sl_batch_img = gr.Slider(1, 4, value=2, step=1, label="Batch Size")
                    sl_steps_img = gr.Slider(10, 50, value=25, label="Steps")
                    sl_cfg_img = gr.Slider(1, 15, value=7, label="CFG")
                    seed_img = gr.Number(value=-1, label="Seed", precision=0)

                btn_gen_img = gr.Button("🎨 GENERA (SD)", variant="primary")

            # Output Destra
            with gr.Column(scale=2):
                with gr.Group():
                    prompt_img = gr.Textbox(label="Prompt Generato", lines=3, interactive=True)
                    neg_img = gr.Textbox(label="Negative", value=DEFAULT_NEGATIVE, lines=2, interactive=True)
                    base_debug = gr.Textbox(visible=False)
                
                gallery_img = gr.Gallery(label="Risultati", columns=2, height="auto")

    # --- TAB 2: DAL TESTO (NUOVO) ---
    with gr.Tab("💡 Creative Text Studio"):
        with gr.Row():
            # Input Sinistra
            with gr.Column(scale=1):
                txt_idea = gr.Textbox(label="La tua Idea", placeholder="es. A cybernetic warrior cat...")
                style_sel_txt = gr.Dropdown(list(STYLES.keys()), value="Cyberpunk", label="Stile")
                
                with gr.Accordion("⚙️ Impostazioni Prompt (GPT-2)", open=True):
                    sl_temp = gr.Slider(0.5, 1.5, value=0.9, label="Creatività (Temperatura)", info="Basso=Preciso, Alto=Folle")
                    sl_len = gr.Slider(50, 200, value=100, step=10, label="Lunghezza Massima")
                
                btn_expand = gr.Button("✨ 1. Espandi Idea", variant="secondary")
                
                gr.Markdown("---")
                with gr.Accordion("Parametri SD", open=False):
                    sl_batch_txt = gr.Slider(1, 4, value=2, step=1, label="Batch Size")
                    sl_steps_txt = gr.Slider(10, 50, value=25, label="Steps")
                    sl_cfg_txt = gr.Slider(1, 15, value=7, label="CFG")
                    seed_txt = gr.Number(value=-1, label="Seed", precision=0)
                
                btn_gen_txt = gr.Button("🎨 GENERA (SD)", variant="primary")

            # Output Destra
            with gr.Column(scale=2):
                with gr.Group():
                    prompt_txt = gr.Textbox(label="Prompt Finale", lines=3, interactive=True)
                    neg_txt = gr.Textbox(label="Negative", value=DEFAULT_NEGATIVE, lines=2, interactive=True)
                
                gallery_txt = gr.Gallery(label="Risultati", columns=2, height="auto")
                lbl_status = gr.Label(show_label=False)

    # --- EVENTI TAB 1 (IMMAGINE) ---
    btn_analyze.click(
        analyze_image_logic, 
        inputs=[img_input, style_sel_img, gr.Checkbox(value=True, visible=False)], 
        outputs=[base_debug, prompt_img, neg_img]
    )
    btn_gen_img.click(
        send_to_sd_api,
        inputs=[prompt_img, neg_img, sl_steps_img, sl_cfg_img, sl_batch_img, seed_img],
        outputs=[gallery_img, lbl_status]
    )

    # --- EVENTI TAB 2 (TESTO) ---
    btn_expand.click(
        text_expansion_logic,
        inputs=[txt_idea, style_sel_txt, sl_temp, sl_len],
        outputs=[prompt_txt, neg_txt]
    )
    btn_gen_txt.click(
        send_to_sd_api,
        inputs=[prompt_txt, neg_txt, sl_steps_txt, sl_cfg_txt, sl_batch_txt, seed_txt],
        outputs=[gallery_txt, lbl_status]
    )

if __name__ == "__main__":
    # Usa una porta sicura
    app.launch(server_name="127.0.0.1", server_port=7863)