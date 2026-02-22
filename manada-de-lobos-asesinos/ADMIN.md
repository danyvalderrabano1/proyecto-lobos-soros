# ADMIN — Instrucciones administrativas

Este documento explica cómo solicitar o configurar un modelo por defecto (por ejemplo "gpt-5-mini") para clientes en tu organización. Nota: este repositorio solo contiene documentación y scripts; no cambia configuraciones de plataforma remotas.

1) Recomendación general
- Contacta al administrador de la plataforma (p.ej., administrador de OpenAI o admin de tu despliegue interno) para solicitar la habilitación del modelo deseado.

2) Configuración local/cliente
- Para aplicaciones que usan variables de entorno, establece una variable por defecto:

```
OPENAI_DEFAULT_MODEL=gpt-5-mini
```

- En tus clientes, lee `process.env.OPENAI_DEFAULT_MODEL` (Node/Python equivalente) y pásalo como `model` a la librería cliente.

3) Ejemplo de configuración (Node.js pseudo-config)

```
const defaultModel = process.env.OPENAI_DEFAULT_MODEL || 'gpt-4o-mini';
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY, defaultModel });
```

4) Permisos y seguridad
- Asegúrate de que el acceso al modelo cumple políticas de uso de datos de tu organización.

5) Solicitar a la plataforma
- Si tu organización usa un panel de control, habilita el modelo allí o pide al proveedor que lo habilite para tu tenant.

6) Qué puedo hacer por ti
- Puedo generar scripts de despliegue, plantillas de CI/CD, o un ejemplo de cliente que use `gpt-5-mini`. Dime si quieres que lo agregue.
