# check_models.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Muat API key dari file .env Anda
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY tidak ditemukan di file .env Anda.")
else:
    try:
        genai.configure(api_key=api_key)

        print("="*50)
        print("Mencari model yang mendukung 'generateContent'...")
        print("="*50)

        # Loop melalui semua model yang tersedia
        for m in genai.list_models():
            # Kita hanya tertarik pada model yang bisa menghasilkan konten (teks/gambar)
            if 'generateContent' in m.supported_generation_methods:
                print(f"Model Name (untuk API): {m.name}")
                print(f"Display Name (Nama): {m.display_name}")
                print(f"Description: {m.description}\n")
        
        print("="*50)
        print("Pencarian selesai.")
        print("Gunakan 'Model Name' di dalam kode Anda.")

    except Exception as e:
        print(f"Terjadi error saat menghubungi Google API: {e}")