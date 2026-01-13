# 1. Alegem imaginea de bază (Sistemul de Operare + Python preinstalat)
# "slim" înseamnă o versiune ușoară, fără chestii inutile.
FROM python:3.14-slim

# 2. Setăm folderul de lucru în interiorul containerului
# De acum, orice comandă rulează în /app
WORKDIR /app

# 3. Copiem fișierul de dependențe
# (Facem asta ÎNAINTE de cod, pentru a folosi cache-ul Docker eficient)
COPY requirements.txt .

# 4. Instalăm librăriile
# --no-cache-dir ține imaginea mică
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiem restul codului nostru (folderul src) în container
COPY src/ src/

# 6. Setăm comanda care rulează când pornește containerul
# Spunem Python-ului unde să caute modulele (folderul curent)
ENV PYTHONPATH=.

# Comanda finală: Rulează scriptul de antrenare
CMD ["python", "src/train.py"]