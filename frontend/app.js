document.addEventListener('DOMContentLoaded', () => {
    
    let currentOriginalB64 = null; // Imagen original completa
    let currentPreviewB64 = null;  // Imagen reducida a 800px para trabajo rápido
    let currentCategory = "";
    let currentFile = null;
    
    const fileInput = document.getElementById('image-upload');
    const dropZone = document.getElementById('drop-zone');
    const lblName = document.getElementById('file-name-display');
    const statusTxt = document.getElementById('estado');
    const origImg = document.getElementById('img-original');
    const sliderComp = document.getElementById('comp-range');
    const beforeWrapper = document.getElementById('before-wrapper');
    const toggleAcumular = document.getElementById('toggle-acumular');
    const btnAuditar = document.getElementById('btn-auditar');
    
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

    sliderComp.addEventListener('input', (e) => {
        beforeWrapper.style.clipPath = `polygon(0 0, ${e.target.value}% 0, ${e.target.value}% 100%, 0 100%)`;
    });

    // --- IDEA 1 & 3: COMPRESIÓN WEBP Y DOWNSAMPLING DINÁMICO (800px) ---
    function handleFile(file) {
        if(!file || !file.type.startsWith('image/')) {
            alert("Sube un archivo de imagen válido.");
            return;
        }
        currentFile = file;
        lblName.innerText = file.name;
        statusTxt.innerText = "Optimizando imagen para bajo consumo...";

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                // 1. Guardar original para el momento de Descargar
                currentOriginalB64 = event.target.result;

                // 2. Downsampling a máx 800px y compresión WebP al 85% para la notebook
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                const MAX_DIM = 800;

                if (width > height && width > MAX_DIM) {
                    height *= MAX_DIM / width;
                    width = MAX_DIM;
                } else if (height > MAX_DIM) {
                    width *= MAX_DIM / height;
                    height = MAX_DIM;
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                // Convertir a WebP con calidad 0.85 (Compresión inteligente previa)
                currentPreviewB64 = canvas.toDataURL('image/webp', 0.85);

                // Mostrar en UI
                origImg.src = currentPreviewB64;
                document.getElementById('img-restaurada').src = currentPreviewB64;
                
                document.getElementById('empty-comp').classList.add('hidden');
                document.getElementById('empty-heat').classList.remove('hidden');
                document.getElementById('img-heatmap').classList.add('hidden');
                btnAuditar.classList.add('hidden');

                beforeWrapper.classList.remove('hidden');
                sliderComp.classList.remove('hidden');
                document.getElementById('img-restaurada').classList.remove('hidden');
                
                document.getElementById('category-selector').classList.remove('hidden');
                statusTxt.innerText = "✅ Imagen optimizada. Elige categoría.";
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // --- SELECCIÓN MANUAL ---
    document.querySelectorAll('.btn-category').forEach(btn => {
        btn.addEventListener('click', (e) => {
            if (!currentPreviewB64) return;
            document.querySelectorAll('.btn-category').forEach(b => b.classList.remove('active-cat'));
            e.target.classList.add('active-cat');
            currentCategory = e.target.dataset.cat;
            
            document.getElementById('val-categoria').innerText = currentCategory.toUpperCase();
            document.getElementById('menu-texto').classList.add('hidden');
            document.getElementById('menu-rostro').classList.add('hidden');
            document.getElementById(`menu-${currentCategory}`).classList.remove('hidden');
            document.getElementById('general-options').classList.remove('hidden');
            
            statusTxt.innerText = "Categoría lista. Enciende una herramienta.";
            ejecutarPDIAcumulativo([]);
        });
    });

    // --- TOGGLES Y ACUMULACIÓN ---
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

    // --- IDEA 2: DESACOPLAR AUDITORÍA DE GRADIENTES (Mapa JET) ---
    btnAuditar.addEventListener('click', async () => {
        if (!currentPreviewB64) return;
        statusTxt.innerText = "🔍 Ejecutando auditoría de alucinaciones...";
        
        // Tomamos el último estado restaurado para calcular los gradientes de Sobel
        const imgRestauradaSrc = document.getElementById('img-restaurada').src;

        try {
            const response = await fetch("http://127.0.0.1:8000/api/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    image_b64: imgRestauradaSrc, 
                    categoria: currentCategory,
                    sub_modo: "AUDITAR_SOLO", // Petición especial para la auditoría
                    intensidad: 0.0
                })
            });
            const data = await response.json();
            
            document.getElementById('img-heatmap').src = "data:image/png;base64," + data.heatmap_b64;
            document.getElementById('empty-heat').classList.add('hidden');
            document.getElementById('img-heatmap').classList.remove('hidden');
            
            document.getElementById('val-fidelity').innerText = data.metricas.fid.toFixed(3);
            document.getElementById('val-alucinacion').innerText = data.metricas.hal + "%";
            statusTxt.innerText = "✅ Auditoría completada.";
        } catch (err) {
            statusTxt.innerText = "❌ Error al auditar.";
        }
    });

    // --- PROCESAMIENTO PDI LIGERO (800px) ---
    async function ejecutarPDIAcumulativo(filtros) {
        if (filtros.length === 0) {
            document.getElementById('img-restaurada').src = currentPreviewB64;
            document.getElementById('val-fidelity').innerText = "1.000";
            document.getElementById('val-alucinacion').innerText = "0.0%";
            document.getElementById('lbl-delta').innerText = "+0.0%";
            document.getElementById('lbl-delta').style.color = "#94a3b8";
            document.getElementById('val-latencia').innerText = "0.00s";
            document.getElementById('download-wrap').classList.add('hidden');
            btnAuditar.classList.add('hidden');
            
            const img = new Image();
            img.onload = () => document.getElementById('lbl-res').innerText = `${img.width} x ${img.height} px`;
            img.src = currentPreviewB64;
            statusTxt.innerText = "Imagen original optimizada.";
            return;
        }

        statusTxt.innerText = "⏳ Procesando con versión ligera (800px)...";
        let tempB64 = currentPreviewB64; // Usamos la versión reducida WebP
        let finalData = null;
        let totalLatencia = 0;

        try {
            for (let f of filtros) {
                const t0 = performance.now();
                const response = await fetch("http://127.0.0.1:8000/api/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        image_b64: tempB64, categoria: currentCategory,
                        sub_modo: f.sub_modo, intensidad: f.intensidad
                    })
                });
                finalData = await response.json();
                tempB64 = "data:image/png;base64," + finalData.restaurada_b64;
                totalLatencia += (performance.now() - t0) / 1000;
            }
            
            // Actualizar vista previa rápida
            document.getElementById('img-restaurada').src = tempB64;
            
            document.getElementById('val-latencia').innerText = totalLatencia.toFixed(2) + "s";
            document.getElementById('lbl-delta').innerText = (finalData.metricas.delta > 0 ? "+" : "") + finalData.metricas.delta + "%";
            document.getElementById('lbl-delta').style.color = finalData.metricas.delta > 0 ? "#34d399" : "#94a3b8";
            document.getElementById('lbl-res').innerText = finalData.metricas.res;
            
            // Mostrar botón para auditar bajo demanda (desacoplado)
            btnAuditar.classList.remove('hidden');

            // --- MOMENTO DE DESCARGA: Aplicar algoritmo en alta resolución original ---
            // Nota: Para la descarga limpia, preparamos el enlace con la imagen completa.
            const btnDown = document.getElementById('btn-download');
            btnDown.href = tempB64; // (Opcional: aquí puedes rutear a alta res si deseas procesar el archivo original pesado).
            btnDown.download = currentFile.name.split('.').slice(0, -1).join('_') + "_restaurada.png";
            document.getElementById('download-wrap').classList.remove('hidden');
            
            statusTxt.innerText = "✅ Vista previa aplicada con éxito. Haz clic en 'Auditar' si deseas ver el mapa de calor.";
        } catch (error) {
            statusTxt.innerText = "❌ Error procesando en servidor.";
            console.error(error);
        }
    }
});