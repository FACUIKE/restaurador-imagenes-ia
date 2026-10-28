document.addEventListener('DOMContentLoaded', () => {
    
    let currentB64 = null;
    let currentCategory = "";
    let currentFile = null;
    
    const fileInput = document.getElementById('image-upload');
    const dropZone = document.getElementById('drop-zone');
    const lblName = document.getElementById('file-name-display');
    const btnAnalizar = document.getElementById('btn-analizar');
    const statusTxt = document.getElementById('estado');
    const origImg = document.getElementById('img-original');
    
    const sliderComp = document.getElementById('comp-range');
    const beforeWrapper = document.getElementById('before-wrapper');
    const toggleAcumular = document.getElementById('toggle-acumular');
    
    // Generar Partículas Fizzy
    const pBox = document.getElementById('particles-box');
    let pCSS = "<style>";
    for(let i=1; i<=40; i++) {
        pBox.innerHTML += '<div class="spots"></div>';
        pCSS += `.fizzy-wrap input:checked ~ .fizzy-label .spots:nth-child(${i}) { transform: translate(${Math.random()*160-80}px, ${Math.random()*160-80}px) scale(0); background: hsl(${Math.floor(Math.random()*40)+180}, 100%, 60%); opacity: 1; transition: transform 0.6s cubic-bezier(0.1, 0.8, 0.2, 1), opacity 0.6s ease-out; } `;
    }
    pCSS += "</style>";
    document.head.insertAdjacentHTML("beforeend", pCSS);

    document.getElementById('fizzy-cb').addEventListener('change', (e) => {
        if(e.target.checked) setTimeout(() => { e.target.checked = false; }, 3000);
    });

    // --- Comparador Deslizante (Cero llamadas a la API) ---
    sliderComp.addEventListener('input', (e) => {
        beforeWrapper.style.clipPath = `polygon(0 0, ${e.target.value}% 0, ${e.target.value}% 100%, 0 100%)`;
    });

    // --- MANEJO DE ARCHIVOS ---
    function handleFile(file) {
        if(!file || !file.type.startsWith('image/')) {
            alert("Sube un archivo de imagen válido (JPG, PNG, WEBP).");
            return;
        }
        
        currentFile = file;
        lblName.innerText = file.name;
        statusTxt.innerText = "Imagen cargada. Lista para analizar.";
        
        document.getElementById('menu-texto').classList.add('hidden');
        document.getElementById('menu-rostro').classList.add('hidden');
        document.getElementById('general-options').classList.add('hidden');
        document.getElementById('download-wrap').classList.add('hidden');
        
        document.querySelectorAll('.tool-toggle').forEach(t => {
            t.checked = false;
            t.closest('.tool-card').classList.remove('active-tool');
        });
        document.querySelectorAll('.slider-box').forEach(s => s.classList.add('hidden'));
        
        btnAnalizar.disabled = false;
        
        const reader = new FileReader();
        reader.onload = (e) => { 
            currentB64 = e.target.result; 
            origImg.src = currentB64; 
            document.getElementById('img-restaurada').src = currentB64;
            document.getElementById('img-heatmap').src = currentB64;

            document.getElementById('empty-comp').classList.add('hidden');
            document.getElementById('empty-heat').classList.add('hidden');

            beforeWrapper.classList.remove('hidden');
            sliderComp.classList.remove('hidden');
            document.getElementById('img-restaurada').classList.remove('hidden');
            document.getElementById('img-heatmap').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
            fileInput.files = e.dataTransfer.files; 
        }
    });

    // --- CLASIFICADOR IA ---
    btnAnalizar.addEventListener('click', async () => {
        if (!currentB64) return alert("Sube una imagen primero.");
        statusTxt.innerText = "Analizando con Inteligencia Artificial...";
        try {
            const response = await fetch("http://localhost:8000/api/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_b64: currentB64 })
            });
            const data = await response.json();
            currentCategory = data.categoria;
            
            document.getElementById('val-categoria').innerText = data.categoria.toUpperCase();
            document.getElementById('val-latencia').innerText = data.latencia.toFixed(2) + "s";
            
            document.getElementById(`menu-${data.categoria}`).classList.remove('hidden');
            document.getElementById('general-options').classList.remove('hidden');
            
            ejecutarPDIAcumulativo([]); // Reset UI
            statusTxt.innerText = "Análisis completado. Enciende una herramienta.";
        } catch (error) {
            statusTxt.innerText = "Error contactando al servidor.";
        }
    });

    // --- LÓGICA DE TOGGLES Y ACUMULACIÓN ---
    
    // Si apagan el modo acumulativo, dejamos solo el primer filtro encendido
    toggleAcumular.addEventListener('change', (e) => {
        if (!e.target.checked) {
            let primero = true;
            document.querySelectorAll('.tool-toggle:checked').forEach(t => {
                if(primero) { primero = false; } 
                else {
                    t.checked = false;
                    t.closest('.tool-card').classList.remove('active-tool');
                    document.getElementById(t.dataset.slider).classList.add('hidden');
                }
            });
            procesarFiltrosActivos();
        }
    });

    const toggles = document.querySelectorAll('.tool-toggle');
    toggles.forEach(toggle => {
        toggle.addEventListener('change', (e) => {
            const card = e.target.closest('.tool-card');
            
            if (e.target.checked) {
                card.classList.add('active-tool');
                
                // Exclusión Mutua SI el botón de acumular está apagado
                if (!toggleAcumular.checked) {
                    toggles.forEach(t => {
                        if (t !== e.target) {
                            t.checked = false;
                            t.closest('.tool-card').classList.remove('active-tool');
                            document.getElementById(t.dataset.slider).classList.add('hidden');
                        }
                    });
                }
                document.getElementById(e.target.dataset.slider).classList.remove('hidden');
            } else {
                card.classList.remove('active-tool');
                document.getElementById(e.target.dataset.slider).classList.add('hidden');
            }
            procesarFiltrosActivos();
        });
    });

    // Escuchar solo a los sliders de herramientas (ignora el comparador)
    document.querySelectorAll('.slider-box input[type="range"]').forEach(slider => {
        slider.addEventListener('change', () => procesarFiltrosActivos());
    });

    function procesarFiltrosActivos() {
        const activeToggles = document.querySelectorAll('.tool-toggle:checked');
        const filtros = [];
        activeToggles.forEach(t => {
            const sliderVal = document.getElementById(t.dataset.slider).querySelector('input').value;
            filtros.push({ sub_modo: t.value, intensidad: parseFloat(sliderVal) });
        });
        ejecutarPDIAcumulativo(filtros);
    }

    // Procesa un array de filtros en cadena
    async function ejecutarPDIAcumulativo(filtros) {
        if (filtros.length === 0) {
            document.getElementById('img-restaurada').src = currentB64;
            document.getElementById('img-heatmap').src = currentB64;
            document.getElementById('val-fidelity').innerText = "1.000";
            document.getElementById('val-alucinacion').innerText = "0.0%";
            document.getElementById('lbl-delta').innerText = "+0.0%";
            document.getElementById('lbl-delta').style.color = "#94a3b8";
            
            const img = new Image();
            img.onload = () => document.getElementById('lbl-res').innerText = `${img.width} x ${img.height} px`;
            img.src = currentB64;
            
            document.getElementById('download-wrap').classList.add('hidden');
            statusTxt.innerText = "Imagen original.";
            return;
        }

        statusTxt.innerText = filtros.length > 1 ? "Aplicando cadena de filtros..." : "Aplicando filtro...";
        
        let tempB64 = currentB64;
        let finalData = null;

        try {
            // Ciclo de peticiones para acumular filtros sin tocar el servidor
            for (let f of filtros) {
                const response = await fetch("http://localhost:8000/api/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        image_b64: tempB64, categoria: currentCategory,
                        sub_modo: f.sub_modo, intensidad: f.intensidad
                    })
                });
                finalData = await response.json();
                tempB64 = "data:image/png;base64," + finalData.restaurada_b64;
            }
            
            // Actualizar Interfaz con el último resultado
            document.getElementById('img-restaurada').src = tempB64;
            document.getElementById('img-heatmap').src = "data:image/png;base64," + finalData.heatmap_b64;
            
            document.getElementById('val-fidelity').innerText = finalData.metricas.fid.toFixed(3);
            document.getElementById('val-alucinacion').innerText = finalData.metricas.hal + "%";
            document.getElementById('lbl-delta').innerText = (finalData.metricas.delta > 0 ? "+" : "") + finalData.metricas.delta + "%";
            document.getElementById('lbl-delta').style.color = finalData.metricas.delta > 0 ? "#34d399" : "#94a3b8";
            document.getElementById('lbl-res').innerText = finalData.metricas.res;
            
            const btnDown = document.getElementById('btn-download');
            btnDown.href = tempB64;
            btnDown.download = currentFile.name.split('.').slice(0, -1).join('.') + "_restaurada.png";
            document.getElementById('download-wrap').classList.remove('hidden');
            
            statusTxt.innerText = filtros.length > 1 ? "Modo Acumulativo Exitoso." : "Filtro aplicado.";
        } catch (error) {
            statusTxt.innerText = "Error procesando imagen.";
        }
    }
});