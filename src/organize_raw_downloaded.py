import os
import shutil
from pathlib import Path
import uuid
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('src/' + Path(__file__).stem + '.log'),
        logging.StreamHandler()
    ]
)

def reorganize_emails_detailed(source_dir, target_dir):
    """
    Versão que inclui informação do nome original no arquivo final.
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Cria diretórios de destino
    ham_target = target_path / 'ham'
    spam_target = target_path / 'spam'
    ham_target.mkdir(parents=True, exist_ok=True)
    spam_target.mkdir(parents=True, exist_ok=True)
    
    # Processa arquivos
    for category in ['ham', 'spam']:
        category_source = source_path / category
        category_target = ham_target if category == 'ham' else spam_target
        count = 0
        
        logging.info(f"Iniciando processamento de arquivos {category.upper()}")
        
        for file_path in category_source.rglob('*'):
            if file_path.is_file():
                # Pega o nome base do arquivo original (sem caminho)
                original_name = file_path.name
                # Gera nome único com info do original
                unique_name = f"{category}_{uuid.uuid4().hex}_{original_name}"
                
                # Remove caracteres problemáticos se houver
                unique_name = unique_name.replace('/', '_').replace('\\', '_')

                if unique_name[-4:] not in ['.txt', '.eml']:
                    unique_name = unique_name + '.txt'
                
                # Cria caminho de destino
                target_file = category_target / unique_name
                
                # Copia o arquivo
                shutil.copy2(file_path, target_file)
                count += 1
        
        logging.info(f"Total de arquivos {category} processados: {count}")

def main():
    reorganize_emails_detailed("data/raw_downloaded/en", "data/raw/en")

if __name__ == "__main__":
    main()
