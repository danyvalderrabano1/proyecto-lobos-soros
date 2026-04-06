Manada de Lobos Asesinos — Dominadores de la Bolsa Soro

Breve scaffold de proyecto y documentación administrativa.

Quick start
- Open this folder in VS Code.
- See `ADMIN.md` for administrative steps (including guidance to set a default model).
- Run the helper script to create the same structure locally: `create_workspace.ps1`.

Project layout
- `src/` — ejemplo de código mínimo
- `.vscode/` — ajustes recomendados para VS Code
- `ADMIN.md` — instrucciones administrativas y seguridad

Notes
- This repository scaffolds a workspace — it does not change platform settings for you. See `ADMIN.md` for steps to request or configure model defaults like "gpt-5-mini" in your environment.

## EA para EURUSD (MetaTrader 5)

Se agregó un ejemplo educativo en `src/EURUSD_TrendEA.mq5` con estas reglas:
- Opera **exclusivamente EURUSD**.
- Confirmación de tendencia en M15 (EMA 50/200).
- Entrada en M5 (o M15) por ruptura de rango de 20 velas.
- Stop Loss y Take Profit por múltiplos de ATR.
- Gestión monetaria por riesgo porcentual (por defecto 1% por trade).

### ¿Se puede convertir USD 1,000 en USD 2,500 al año?
No se puede garantizar. Pasar de 1,000 a 2,500 implica un retorno de **150% anual**, que es muy alto y conlleva riesgo significativo.

Lo correcto es:
1. Backtest robusto (mínimo 5-10 años de datos EURUSD).
2. Walk-forward y pruebas fuera de muestra.
3. Prueba en demo antes de cuenta real.
4. Definir límite de drawdown máximo aceptable.
