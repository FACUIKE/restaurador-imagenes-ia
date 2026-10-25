from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test_conexion():
    data = request.get_json()
    print("[+] ¡Conexión exitosa recibida desde n8n!")
    print(f"[+] Datos recibidos: {data}")
    return jsonify({
        "status": "ok",
        "mensaje": "¡Hola n8n! La PC se conectó correctamente."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)