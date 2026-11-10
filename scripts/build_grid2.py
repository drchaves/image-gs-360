from PIL import Image, ImageDraw, ImageFont
import os

def process_grid():
    # Paths
    base = "results/"
    ext = "/num-100000_inv-scale-5.0_bits-14-14-14-14_top-10_g-0.3_l1-1.0_l2-0.0_ssim-0.1_decay-1-10.0_prog"
    
    paths = {
        'photo_gt': f"{base}exp_photo-5_2k_g100000_L0.0{ext}/gt_res-2048x2048.jpg",
        'photo_anc': f"{base}exp_photo-5_2k_g100000_L0.0{ext}/render_res-2048x2048.jpg",
        'photo_rdo': f"{base}exp_photo-5_2k_g100000_L0.0005{ext}/render_res-2048x2048.jpg",
        'art_gt': f"{base}exp_art-1_2k_g100000_L0.0{ext}/gt_res-2048x2048.jpg",
        'art_anc': f"{base}exp_art-1_2k_g100000_L0.0{ext}/render_res-2048x2048.jpg",
        'art_rdo': f"{base}exp_art-1_2k_g100000_L0.0005{ext}/render_res-2048x2048.jpg"
    }

    crop_size = 600
    
    def get_crop(path):
        img = Image.open(path)
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return img.crop((left, top, left + crop_size, top + crop_size))

    crops = {k: get_crop(v) for k, v in paths.items()}

    header_height = 80
    sidebar_width = 80
    margin = 10
    
    grid_w = sidebar_width + 3 * crop_size + 4 * margin
    grid_h = header_height + 2 * crop_size + 3 * margin
    
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        
    headers = ["Original (Referência)", "Image-GS (Âncora)", "GaussianARM (Proposto)"]
    for i, text in enumerate(headers):
        x = sidebar_width + margin + i * (crop_size + margin) + crop_size // 2
        y = header_height // 2
        draw.text((x, y), text, fill="black", font=font, anchor="mm")

    sidebars = ["Fotografia", "Arte"]
    for i, text in enumerate(sidebars):
        x = sidebar_width // 2
        y = header_height + margin + i * (crop_size + margin) + crop_size // 2
        
        txt_img = Image.new('RGB', (200, sidebar_width), 'white')
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((100, sidebar_width//2), text, fill="black", font=font, anchor="mm")
        txt_img = txt_img.rotate(90, expand=1)
        
        paste_y = y - 100
        grid.paste(txt_img, (0, paste_y))

    layout = [
        ['photo_gt', 'photo_anc', 'photo_rdo'],
        ['art_gt', 'art_anc', 'art_rdo']
    ]
    
    for row, keys in enumerate(layout):
        for col, key in enumerate(keys):
            x = sidebar_width + margin + col * (crop_size + margin)
            y = header_height + margin + row * (crop_size + margin)
            grid.paste(crops[key], (x, y))

    out_path = os.path.join("tcc_diego", "figuras", "comparacao_visual.png")
    grid.save(out_path)
    print(f"Salvo em: {out_path}")

if __name__ == "__main__":
    process_grid()
