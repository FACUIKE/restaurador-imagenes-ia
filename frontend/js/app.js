document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnProcess = document.getElementById("btn-process");
  const fileInfo = document.getElementById("file-info");
  const infoName = document.getElementById("info-name");
  const infoRes = document.getElementById("info-res");
  const btnDownload = document.getElementById("btn-download");

  // Elementos del visor
  const emptyComp = document.getElementById("empty-state-comp");
  const emptyHeat = document.getElementById("empty-state-heat");
  const imgBefore = document.getElementById("img-before");
  const imgAfter = document.getElementById("img-after");
  const beforeWrapper = document.getElementById("before-wrapper");
  const sliderLine = document.getElementById("slider-line");
  const compRange = document.getElementById("comp-range");
  const imgHeatmap = document.getElementById("img-heatmap");
  const heatCaption = document.getElementById("heat-caption");

  // Elementos de métricas
  const valCategoria = document.getElementById("val-categoria");
  const valLatencia = document.getElementById("val-latencia");
  const valFidelity = document.getElementById("val-fidelity");
  const valAlucinacion = document.getElementById("val-alucinacion");
  const sideSsim = document.getElementById("side-ssim");
  const sideNitidez = document.getElementById("side-nitidez");
  const sideRes = document.getElementById("side-res");

  let selectedFile = null;

  // Función común para cargar la imagen
  function handleFileSelection(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Por favor selecciona un archivo de imagen válido.");
      return;
    }

    selectedFile = file;
    infoName.innerHTML = `📄 <b>Archivo:</b> ${file.name}`;

    const tempImg = new Image();
    tempImg.src = URL.createObjectURL(file);
    tempImg.onload = () => {
      infoRes.innerHTML = `📐 <b>Resolución:</b> ${tempImg.naturalWidth} x ${tempImg.naturalHeight} px`;
    };

    fileInfo.classList.remove("hidden");
    btnProcess.disabled = false;
  }

  // 1. Clic en la zona abre el selector de archivo
  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  // 2. Drag & Drop nativo
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  // 3. Movimiento del deslizador interactivo
  compRange.addEventListener("input", (e) => {
    const val = e.target.value;
    // En lugar de achicar el ancho, usamos una máscara matemática de recorte (clip-path)
    beforeWrapper.style.clipPath = `polygon(0 0, ${val}% 0, ${val}% 100%, 0 100%)`;
    sliderLine.style.left = `${val}%`;
  });

  // 4. Envío al backend
  btnProcess.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnProcess.disabled = true;
    btnProcess.textContent = "⏳ Procesando en RAM...";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Error en la inferencia del servidor.");

      const data = await response.json();

      // Actualizar Telemetría Superior
      valCategoria.textContent = data.categoria;
      valLatencia.textContent = `${data.latencia}s`;
      valFidelity.textContent = Number(data.metricas.fidelity_score).toFixed(3);
      valAlucinacion.textContent = `${data.metricas.hallucination_ratio}%`;

      // Actualizar Tarjetas Laterales
      sideSsim.textContent = Number(data.metricas.ssim).toFixed(4);
      sideNitidez.textContent = `+${data.metricas.delta_nitidez}%`;
      sideRes.textContent = data.metricas.out_res;

      // Renderizar Comparador Antes / Después
      imgBefore.src = data.img_original;
      imgAfter.src = data.img_restaurada;
      emptyComp.classList.add("hidden");
      imgAfter.classList.remove("hidden");
      beforeWrapper.classList.remove("hidden");
      sliderLine.classList.remove("hidden");

      // Renderizar Mapa de Calor de Gradientes
      imgHeatmap.src = data.img_heatmap;
      emptyHeat.classList.add("hidden");
      imgHeatmap.classList.remove("hidden");
      heatCaption.classList.remove("hidden");

      // Habilitar Descarga
      btnDownload.href = data.img_restaurada;
      btnDownload.classList.remove("hidden");

    } catch (err) {
      alert("Hubo un error al procesar la imagen: " + err.message);
    } finally {
      btnProcess.disabled = false;
      btnProcess.textContent = "⚡ Ejecutar Restauración + Auditoría";
    }
  });
});