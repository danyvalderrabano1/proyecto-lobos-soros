# Inicializa cliente usando variable de entorno
client = OpenAI()

def lobo_model(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Eres un modelo de IA especializado en trading algorítmico y análisis cuantitativo."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def main():
    print("=== LOBO IA ENGINE ===")
    user_input = input("Haz tu pregunta de trading: ")
    respuesta = lobo_model(user_input)
    print("\nRespuesta IA:\n")
    print(respuesta)


if _name_ == "_main_":
    main()