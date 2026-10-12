import sys
import subprocess
from app.frontend import render_app

def is_running_inside_streamlit() -> bool:
    """Détecte si le script est actuellement exécuté via le serveur Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def main():
    if is_running_inside_streamlit():
        # Exécution normale à l'intérieur du runtime Streamlit
        render_app()
    else:
        # Si l'utilisateur a tapé 'python main.py' ou 'uv run main.py',
        # on lance automatiquement Streamlit sur ce même fichier !
        print(" Lancement de l'interface Streamlit via main.py...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

if __name__ == "__main__":
    main()
