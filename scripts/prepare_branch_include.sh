#!/usr/bin/env bash
# prepare_branch_include.sh
# Crea una nueva rama e opcionalmente modifica .gitignore para incluir archivos específicos
# Uso: ./scripts/prepare_branch_include.sh <branch-name> [paths-to-include-file]

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GITROOT="$ROOT_DIR"
IGNOREF="$GITROOT/.gitignore"
BRANCH="${1:-include-artifacts}"
INCLUDE_FILE="${2:-}"  # opcion: archivo con una lista de paths que quieres forzar commit (one per line)

if [ -z "$BRANCH" ]; then
  echo "Uso: $0 <branch-name> [include-list.txt]"
  exit 1
fi

cd "$GITROOT"

echo "Creando rama: $BRANCH"
git checkout -b "$BRANCH"

if [ -n "$INCLUDE_FILE" ] && [ -f "$INCLUDE_FILE" ]; then
  echo "Procesando include list: $INCLUDE_FILE"
  # para cada ruta en include file, quitar ignore si está en .gitignore o aplicar negate (!path)
  while IFS= read -r path || [ -n "$path" ]; do
    [ -z "$path" ] && continue
    # si .gitignore contiene la ruta, comentar la linea o añadir !ruta
    if grep -qE "^${path}$" "$IGNOREF" 2>/dev/null; then
      echo "Negating $path in .gitignore"
      sed -i.bak "s|^${path}$|!${path}|" "$IGNOREF"
    else
      # añadir excepción
      echo "!${path}" >> "$IGNOREF"
    fi
  done < "$INCLUDE_FILE"
  echo ".gitignore actualizado con excepciones definidos en $INCLUDE_FILE"
  echo "Nota: revisa .gitignore antes de commitear"
else
  echo "No se proporcionó include list. Si quieres incluir rutas específicas, crea un archivo con paths y pásalo como segundo argumento"
fi

echo "Ahora puedes añadir y commitear archivos grandes seleccionados. Ej: git add -f uploads/gestures/* && git commit -m 'add gestures'"

echo "Cuando termines, recuerda revertir cambios en .gitignore o eliminar las excepciones y push la rama: git push -u origin $BRANCH"
