document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Variables Globales y Selectores ---
    let currentB64 = null;
    let currentCategory = "";
    
    const fileInput = document.getElementById('image-upload');
    const lblName = document.getElementById('file-name-display');
    const btnAnalizar = document.getElementById('btn-analizar');
    const statusTxt = document.getElementById('estado');
    
    const sliderComp = document.getElementById('comp-range');
    const beforeWrapper = document.getElementById('before-wrapper');
    
    // Generar Partículas Fizzy
    const pBox = document.getElementById('particles-box');
    let pCSS = "<style>";
    for(let i=1; i<=40; i++) {
        pBox.innerHTML += '<div class="spots"></div>';
        pCSS += `.fizzy-wrap input:checked ~ .fizzy-label .spots:nth-child(${i}) { transform: translate(${Math.random()*160-80}px, ${Math.random()*160-80}px) scale(0); background: hsl(${Math.floor(Math.random()*40)+180}, 100%, 60%); opacity: 1; transition: transform 0.6s cubic-bezier(0.1, 0.8, 0.2, 1), opacity 0.6s ease-out; } `;
    }
    pCSS += "</style>";
    document.head.insertAdjacentHTML("beforeend", pCSS);

    // Animación Botón Fizzy Reset
    document.getElementById('fizzy-cb').addEventListener('change', (e) => {
        if(e.target.checked) setTimeout(() => { e.target.checked = false; }, 3000);
    });

    // --- 2. Comparador Deslizante ---
    sliderComp.addEventListener('input', (e) => {
        beforeWrapper.style.clipPath = `polygon(0 0, ${e.target.value}% 0, ${e.target.value}% 100%, 0 100%)`;
    });

    // --- 3. Lector de Archivos a Base64 (Evita Form-Data) ---
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if(!file) return;
        lblName.innerText = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => { currentB64 = e.target.result; };
        reader.readAsDataURL(file);
    });

    // --- 4. Enviar a Clasificar ---
    btnAnalizar.addEventListener('click', async () => {
        if (!currentB64) return alert("Sube una imagen y espera que se cargue.");
        
        statusTxt.innerText = "Clasificando con IA...";
        
        try {
            const response = await fetch("http://localhost:8000/api/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_b64: currentB64 })
            });
            
            const data = await response.json();
            currentCategory = data.categoria;
            
            // UI Update
            document.getElementById('val-categoria').innerText = data.categoria.toUpperCase();
            document.getElementById('val-latencia').innerText = data.latencia.toFixed(2) + "s";
            
            const origImg = document.getElementById('img-original');
            origImg.src = "data:image/png;base64," + data.original_b64;
            document.getElementById('empty-comp').classList.add('hidden');
            beforeWrapper.classList.remove('hidden');
            sliderComp.classList.remove('hidden');
            
            // Menús Dinámicos
            document.getElementById('menu-texto').classList.add('hidden');
            document.getElementById('menu-rostro').classList.add('hidden');
            document.getElementById(`menu-${data.categoria}`).classList.remove('hidden');
            
            // Resetear Radios
            document.querySelectorAll('input[type="radio"]').forEach(r => {
                r.checked = false;
                r.nextElementSibling.classList.add('hidden'); // Ocultar slider
            });

            statusTxt.innerText = "Listo. Selecciona una herramienta.";
            
        } catch (error) {
            statusTxt.innerText = "Error en el servidor de IA.";
        }
    });

    // --- 5. Lógica de Radio Buttons y Sliders PDI ---
    document.querySelectorAll('input[type="radio"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            // Ocultar todos los sliders y mostrar el activo
            document.querySelectorAll('.slider-box').forEach(sb => sb.classList.add('hidden'));
            const sbox = e.target.nextElementSibling;
            sbox.classList.remove('hidden');
            
            const slider = sbox.querySelector('input[type="range"]');
            
            // Ejecutar procesamiento con el valor inicial
            ejecutarPDI(e.target.value, slider.value);
            
            // Ejecutar procesamiento al mover el slider (On-Change / On-Release)
            slider.onchange = () => ejecutarPDI(e.target.value, slider.value);
        });
    });

    async function ejecutarPDI(subModo, intensidadStr) {
        statusTxt.innerText = "Calculando matemática...";
        const intensidad = parseFloat(intensidadStr);

        try {
            const response = await fetch("http://localhost:8000/api/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    image_b64: currentB64,
                    categoria: currentCategory,
                    sub_modo: subModo,
                    intensidad: intensidad
                })
            });
            
            const data = await response.json();
            
            // Actualizar Imágenes
            const b64_rest = "data:image/png;base64," + data.restaurada_b64;
            document.getElementById('img-restaurada').src = b64_rest;
            document.getElementById('img-restaurada').classList.remove('hidden');
            
            document.getElementById('img-heatmap').src = "data:image/png;base64," + data.heatmap_b64;
            document.getElementById('empty-heat').classList.add('hidden');
            document.getElementById('img-heatmap').classList.remove('hidden');
            
            // Actualizar Métricas
            document.getElementById('val-fidelity').innerText = data.metricas.fid.toFixed(3);
            document.getElementById('val-alucinacion').innerText = data.metricas.hal + "%";
            document.getElementById('lbl-delta').innerText = "+" + data.metricas.delta + "%";
            document.getElementById('lbl-res').innerText = data.metricas.res;
            
            // Habilitar Botón de Descarga
            const btnDown = document.getElementById('btn-download');
            btnDown.href = b64_rest;
            document.getElementById('download-wrap').classList.remove('hidden');
            
            statusTxt.innerText = "Filtro aplicado.";
        } catch (error) {
            statusTxt.innerText = "Error aplicando filtro.";
        }
    }
});