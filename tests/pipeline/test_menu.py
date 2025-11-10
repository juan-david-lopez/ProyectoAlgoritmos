"""
Test del menú interactivo
"""
import subprocess
import sys

print("Iniciando menú interactivo...")
print("Si se cierra automáticamente, hay un error en el código\n")

try:
    subprocess.run([sys.executable, "menu_interactivo.py"], check=True)
except KeyboardInterrupt:
    print("\n\n👋 Saliendo...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
