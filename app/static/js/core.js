const CONFIG = window.SEGIMBUD_CONFIG || {};

export function apiUrl(path) {
    const base = CONFIG.apiBase || "/api";
    return `${base}${path}`;
}

export async function fetchJson(path, options = {}) {
    const response = await fetch(apiUrl(path), options);
    const contentType = response.headers.get("content-type") || "";
    let payload;

    if (contentType.includes("application/json")) {
        payload = await response.json().catch(() => ({}));
    } else {
        payload = await response.text().catch(() => "");
    }

    if (!response.ok) {
        const message =
            typeof payload === "string"
                ? payload
                : payload.detail || payload.error || "Request failed.";
        throw new Error(message);
    }

    return payload;
}

export async function fetchBlob(path, options = {}) {
    const response = await fetch(apiUrl(path), options);
    if (!response.ok) {
        const contentType = response.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.detail || payload.error || "Request failed.");
        }
        throw new Error((await response.text().catch(() => "")) || "Request failed.");
    }
    return response.blob();
}

export async function refreshGlobalHealth() {
    const pill = document.getElementById("global-health-pill");
    if (!pill) {
        return null;
    }

    pill.textContent = "Checking model";
    pill.classList.remove("is-ready", "is-degraded", "is-error");

    try {
        const payload = await fetchJson("/health");
        if (payload.ready) {
            pill.textContent = `Model ready | ${String(payload.device || "unknown").toUpperCase()}`;
            pill.classList.add("is-ready");
        } else {
            pill.textContent = "Model degraded";
            pill.classList.add("is-degraded");
        }
        return payload;
    } catch (error) {
        pill.textContent = "API offline";
        pill.classList.add("is-error");
        return null;
    }
}

export function dataUrlFromBase64(base64) {
    return base64 ? `data:image/png;base64,${base64}` : "";
}

export function formatPercent(value) {
    return `${Number(value || 0).toFixed(2)}%`;
}

export function formatAreaKm2(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${Number(value).toFixed(6)} km^2`;
}

export function dominantClass(stats) {
    const entries = Object.entries(stats || {});
    entries.sort((a, b) => (b[1]?.pixels || 0) - (a[1]?.pixels || 0));
    return entries.length ? entries[0][0] : "Unknown";
}

export async function loadImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Image failed to load."));
        image.src = src;
    });
}

export async function imageDataFromSource(src) {
    const image = await loadImage(src);
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth || image.width;
    canvas.height = image.naturalHeight || image.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0);
    return {
        image,
        canvas,
        width: canvas.width,
        height: canvas.height,
        imageData: ctx.getImageData(0, 0, canvas.width, canvas.height),
    };
}

export async function imageDataFromBase64(base64) {
    return imageDataFromSource(dataUrlFromBase64(base64));
}

export async function grayscaleArrayFromBase64(base64) {
    const source = await imageDataFromBase64(base64);
    const grayscale = new Uint8ClampedArray(source.width * source.height);
    for (let index = 0; index < grayscale.length; index += 1) {
        grayscale[index] = source.imageData.data[index * 4];
    }
    return {
        width: source.width,
        height: source.height,
        data: grayscale,
    };
}

export function rowsFromStats(stats, resolutionMetersPerPixel = 0) {
    const pixelAreaSqKm = resolutionMetersPerPixel > 0 ? (resolutionMetersPerPixel ** 2) / 1_000_000 : null;
    const rows = Object.entries(stats || {}).map(([name, values]) => ({
        class: name,
        classId: values.class_id,
        pixels: Number(values.pixels || 0),
        percentage: Number(values.percentage || 0),
        color: values.color || [255, 255, 255],
        areaSqKm:
            values.area_sq_km !== undefined
                ? Number(values.area_sq_km)
                : pixelAreaSqKm !== null
                  ? Number(values.pixels || 0) * pixelAreaSqKm
                  : null,
    }));
    rows.sort((a, b) => b.pixels - a.pixels);
    return rows;
}

export function buildLegendMap(legendEntries) {
    return new Map((legendEntries || []).map((entry) => [Number(entry.class_id), entry]));
}

function makeCanvas(width, height) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
}

export function renderMaskDataUrl(mask, legendEntries, visibleClassIds, neutral = [239, 234, 225]) {
    const legend = buildLegendMap(legendEntries);
    const canvas = makeCanvas(mask.width, mask.height);
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(mask.width, mask.height);

    for (let index = 0; index < mask.data.length; index += 1) {
        const classId = mask.data[index];
        const target = index * 4;
        const legendEntry = legend.get(classId);
        const color =
            visibleClassIds.has(classId) && legendEntry ? legendEntry.color : neutral;
        imageData.data[target] = color[0];
        imageData.data[target + 1] = color[1];
        imageData.data[target + 2] = color[2];
        imageData.data[target + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL("image/png");
}

export function renderOverlayDataUrl(original, mask, legendEntries, visibleClassIds, alpha = 0.58) {
    const legend = buildLegendMap(legendEntries);
    const canvas = makeCanvas(original.width, original.height);
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(original.width, original.height);
    const source = original.imageData.data;
    const output = imageData.data;

    for (let index = 0; index < mask.data.length; index += 1) {
        const target = index * 4;
        const classId = mask.data[index];
        if (visibleClassIds.has(classId) && legend.has(classId)) {
            const color = legend.get(classId).color;
            output[target] = Math.round(source[target] * (1 - alpha) + color[0] * alpha);
            output[target + 1] = Math.round(source[target + 1] * (1 - alpha) + color[1] * alpha);
            output[target + 2] = Math.round(source[target + 2] * (1 - alpha) + color[2] * alpha);
        } else {
            output[target] = source[target];
            output[target + 1] = source[target + 1];
            output[target + 2] = source[target + 2];
        }
        output[target + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL("image/png");
}

export function renderUncertaintyDataUrl(original, confidenceMap, threshold = 0.6) {
    const canvas = makeCanvas(original.width, original.height);
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(original.width, original.height);
    const source = original.imageData.data;
    const output = imageData.data;

    for (let index = 0; index < confidenceMap.data.length; index += 1) {
        const target = index * 4;
        const confidence = confidenceMap.data[index] / 255;
        if (confidence < threshold) {
            output[target] = Math.round(source[target] * 0.42 + 214 * 0.58);
            output[target + 1] = Math.round(source[target + 1] * 0.42 + 131 * 0.58);
            output[target + 2] = Math.round(source[target + 2] * 0.42 + 56 * 0.58);
        } else {
            output[target] = source[target];
            output[target + 1] = source[target + 1];
            output[target + 2] = source[target + 2];
        }
        output[target + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL("image/png");
}

export function buildCsv(rows, fieldOrder) {
    const header = `${fieldOrder.join(",")}\n`;
    const lines = rows
        .map((row) =>
            fieldOrder
                .map((field) => {
                    const value = row[field] ?? "";
                    const serialized = String(value).replace(/"/g, '""');
                    return `"${serialized}"`;
                })
                .join(","),
        )
        .join("\n");
    return `${header}${lines}`;
}

export function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
}

export function downloadBase64(filename, base64) {
    const anchor = document.createElement("a");
    anchor.href = dataUrlFromBase64(base64);
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
}

export function setImageTarget(target, src, emptyElement) {
    if (!target) {
        return;
    }
    if (src) {
        target.src = src;
        target.classList.add("is-visible");
        if (emptyElement) {
            emptyElement.classList.add("is-hidden");
        }
    } else {
        target.removeAttribute("src");
        target.classList.remove("is-visible");
        if (emptyElement) {
            emptyElement.classList.remove("is-hidden");
        }
    }
}

export function classIdsFromSelection(values, legendEntries) {
    const nameToId = new Map((legendEntries || []).map((entry) => [entry.name, Number(entry.class_id)]));
    return new Set(values.map((name) => nameToId.get(name)).filter((value) => value !== undefined));
}

export function createToggleMarkup(entry, checked = true) {
    return `
        <label class="toggle-chip">
            <span class="toggle-chip__meta">
                <span class="toggle-chip__swatch" style="background: rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]});"></span>
                <span class="toggle-chip__text">${entry.name}</span>
            </span>
            <input type="checkbox" value="${entry.name}" ${checked ? "checked" : ""}>
        </label>
    `;
}
