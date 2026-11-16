/**
 * NeuraType AI - Next Word Prediction & Text Completion Client
 * Interactive frontend application logic.
 */

(function () {
    "use strict";

    // Application State
    const state = {
        apiBase: "",
        currentPredictions: [],
        topWord: "",
        isPredicting: false,
        isGenerating: false,
        debounceTimer: null,
        autoPredict: true,
        modelLoaded: false,
        theme: localStorage.getItem("neuratype_theme") || "dark"
    };

    // DOM Elements
    const elements = {
        themeToggle: document.getElementById("themeToggle"),
        statusBadge: document.getElementById("statusBadge"),
        statusText: document.getElementById("statusText"),
        statusDot: document.querySelector(".status-dot"),
        textInput: document.getElementById("textInput"),
        ghostOverlay: document.getElementById("ghostOverlay"),
        ghostTyped: document.getElementById("ghostTyped"),
        ghostSuggestion: document.getElementById("ghostSuggestion"),
        tabHintBar: document.getElementById("tabHintBar"),
        ghostWordHighlight: document.getElementById("ghostWordHighlight"),
        btnAcceptGhost: document.getElementById("btnAcceptGhost"),
        chipsContainer: document.getElementById("chipsContainer"),
        probabilitiesList: document.getElementById("probabilitiesList"),
        charCount: document.getElementById("charCount"),
        wordCount: document.getElementById("wordCount"),
        latencyStat: document.getElementById("latencyStat"),
        btnClear: document.getElementById("btnClear"),
        btnCopy: document.getElementById("btnCopy"),
        btnPredict: document.getElementById("btnPredict"),
        autoPredictToggle: document.getElementById("autoPredictToggle"),
        presetChipsContainer: document.getElementById("presetChipsContainer"),
        // Generator controls
        wordCountSlider: document.getElementById("wordCountSlider"),
        wordCountVal: document.getElementById("wordCountVal"),
        tempSlider: document.getElementById("tempSlider"),
        tempVal: document.getElementById("tempVal"),
        genModeSelect: document.getElementById("genModeSelect"),
        btnGenerate: document.getElementById("btnGenerate"),
        genResultBox: document.getElementById("genResultBox"),
        genResultText: document.getElementById("genResultText"),
        btnApplyGen: document.getElementById("btnApplyGen"),
        // Inspector elements
        inspectorToggle: document.getElementById("inspectorToggle"),
        inspectorBody: document.getElementById("inspectorBody"),
        inspectCleaned: document.getElementById("inspectCleaned"),
        inspectTokens: document.getElementById("inspectTokens"),
        inspectTokenIds: document.getElementById("inspectTokenIds"),
        inspectVector: document.getElementById("inspectVector"),
        specVocab: document.getElementById("specVocab"),
        specSize: document.getElementById("specSize"),
        toast: document.getElementById("toast")
    };

    // Initialize Application
    async function init() {
        initTheme();
        bindEvents();
        await checkHealth();
        await loadModelInfo();
        await loadExamples();
        syncStats();
    }

    // Theme Management
    function initTheme() {
        if (state.theme === "light") {
            document.body.classList.remove("dark-theme");
            document.body.classList.add("light-theme");
        } else {
            document.body.classList.add("dark-theme");
            document.body.classList.remove("light-theme");
        }
    }

    function toggleTheme() {
        if (document.body.classList.contains("light-theme")) {
            document.body.classList.replace("light-theme", "dark-theme");
            state.theme = "dark";
        } else {
            document.body.classList.replace("dark-theme", "light-theme");
            state.theme = "light";
        }
        localStorage.setItem("neuratype_theme", state.theme);
    }

    // Health & Model Info
    async function checkHealth() {
        try {
            const res = await fetch(`${state.apiBase}/api/health`);
            const data = await res.json();
            if (data.status === "healthy" && data.model_loaded) {
                elements.statusDot.className = "status-dot online pulse";
                elements.statusText.textContent = "Model Online (Ready)";
                state.modelLoaded = true;
            } else {
                elements.statusDot.className = "status-dot pulse";
                elements.statusText.textContent = "Loading Model...";
                state.modelLoaded = false;
            }
        } catch (err) {
            elements.statusDot.className = "status-dot offline";
            elements.statusText.textContent = "Backend Offline";
            state.modelLoaded = false;
        }
    }

    async function loadModelInfo() {
        try {
            const res = await fetch(`${state.apiBase}/api/info`);
            if (res.ok) {
                const info = await res.json();
                if (info.vocab_size) {
                    elements.specVocab.textContent = info.vocab_size.toLocaleString();
                }
                if (info.weights_size_mb) {
                    elements.specSize.textContent = `${info.weights_size_mb} MB`;
                }
            }
        } catch (e) {
            console.warn("Could not load model metadata", e);
        }
    }

    async function loadExamples() {
        try {
            const res = await fetch(`${state.apiBase}/api/examples`);
            if (res.ok) {
                const examples = await res.json();
                if (examples && examples.length > 0) {
                    elements.presetChipsContainer.innerHTML = "";
                    examples.forEach(ex => {
                        const chip = document.createElement("button");
                        chip.className = "preset-chip";
                        chip.textContent = `${ex.prompt}...`;
                        chip.title = ex.description;
                        chip.dataset.prompt = ex.prompt;
                        elements.presetChipsContainer.appendChild(chip);
                    });
                }
            }
        } catch (e) {
            console.warn("Could not fetch examples, using defaults.");
        }
    }

    // Event Bindings
    function bindEvents() {
        elements.themeToggle.addEventListener("click", toggleTheme);

        // Editor Input
        elements.textInput.addEventListener("input", handleInput);
        elements.textInput.addEventListener("keydown", handleKeyDown);
        elements.textInput.addEventListener("scroll", syncGhostScroll);

        // Auto predict toggle
        elements.autoPredictToggle.addEventListener("change", (e) => {
            state.autoPredict = e.target.checked;
            if (state.autoPredict && elements.textInput.value.trim()) {
                requestPrediction();
            }
        });

        // Predict button
        elements.btnPredict.addEventListener("click", () => {
            requestPrediction(true);
        });

        // Clear button
        elements.btnClear.addEventListener("click", clearEditor);

        // Copy button
        elements.btnCopy.addEventListener("click", copyText);

        // Ghost accept buttons
        elements.btnAcceptGhost.addEventListener("click", acceptGhostWord);

        // Generator controls
        elements.wordCountSlider.addEventListener("input", (e) => {
            elements.wordCountVal.textContent = `${e.target.value} words`;
        });

        elements.tempSlider.addEventListener("input", (e) => {
            elements.tempVal.textContent = e.target.value;
        });

        elements.btnGenerate.addEventListener("click", generateMultiWord);
        elements.btnApplyGen.addEventListener("click", applyGeneratedText);

        // Preset chips click
        elements.presetChipsContainer.addEventListener("click", (e) => {
            const chip = e.target.closest(".preset-chip");
            if (chip && chip.dataset.prompt) {
                elements.textInput.value = chip.dataset.prompt + " ";
                elements.textInput.focus();
                syncStats();
                requestPrediction(true);
            }
        });

        // Inspector drawer toggle
        elements.inspectorToggle.addEventListener("click", () => {
            elements.inspectorToggle.classList.toggle("collapsed");
            elements.inspectorBody.classList.toggle("hidden");
        });
    }

    // Real-Time Input Handling
    function handleInput() {
        syncStats();
        updateGhostOverlay();

        if (!state.autoPredict) return;

        clearTimeout(state.debounceTimer);
        state.debounceTimer = setTimeout(() => {
            requestPrediction();
        }, 280);
    }

    function handleKeyDown(e) {
        // Tab key or Right Arrow (if at end) accepts ghost prediction
        if (state.topWord && (e.key === "Tab" || (e.key === "ArrowRight" && isCursorAtEnd()))) {
            e.preventDefault();
            acceptGhostWord();
        }
    }

    function isCursorAtEnd() {
        const val = elements.textInput.value;
        return elements.textInput.selectionStart === val.length && elements.textInput.selectionEnd === val.length;
    }

    function syncGhostScroll() {
        elements.ghostOverlay.scrollTop = elements.textInput.scrollTop;
    }

    function syncStats() {
        const text = elements.textInput.value;
        elements.charCount.textContent = `${text.length} characters`;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        elements.wordCount.textContent = `${words} words`;
    }

    function updateGhostOverlay() {
        const text = elements.textInput.value;
        elements.ghostTyped.textContent = text;

        if (state.topWord && text.trim().length > 0 && isCursorAtEnd()) {
            const needsSpace = !text.endsWith(" ");
            const ghostDisplay = (needsSpace ? " " : "") + state.topWord;
            elements.ghostSuggestion.textContent = ghostDisplay;
            elements.ghostSuggestion.style.display = "inline";

            elements.ghostWordHighlight.textContent = state.topWord;
            elements.tabHintBar.style.display = "flex";
        } else {
            elements.ghostSuggestion.textContent = "";
            elements.ghostSuggestion.style.display = "none";
            elements.tabHintBar.style.display = "none";
        }
    }

    function acceptGhostWord() {
        if (!state.topWord) return;
        const currentText = elements.textInput.value;
        const needsSpace = !currentText.endsWith(" ");
        elements.textInput.value = currentText + (needsSpace ? " " : "") + state.topWord + " ";
        elements.textInput.focus();
        elements.textInput.setSelectionRange(elements.textInput.value.length, elements.textInput.value.length);
        
        syncStats();
        requestPrediction(true);
    }

    // Prediction Request
    async function requestPrediction(force = false) {
        const text = elements.textInput.value.trim();
        if (!text) {
            clearPredictions();
            return;
        }

        if (state.isPredicting && !force) return;
        state.isPredicting = true;
        elements.btnPredict.classList.add("loading");

        try {
            const res = await fetch(`${state.apiBase}/api/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    top_k: 8,
                    temperature: parseFloat(elements.tempSlider.value) || 1.0
                })
            });

            const data = await res.json();
            if (data.success && data.predictions && data.predictions.length > 0) {
                state.currentPredictions = data.predictions;
                state.topWord = data.predictions[0].word;
                renderChips(data.predictions);
                renderProbabilities(data.predictions);
                renderInspector(data.tokens_info);
                elements.latencyStat.textContent = `Latency: ${data.latency_ms} ms`;
            } else {
                state.currentPredictions = [];
                state.topWord = "";
                elements.chipsContainer.innerHTML = `<span class="chip-placeholder">${data.error || "No predictions found"}</span>`;
                elements.latencyStat.textContent = `Latency: ${data.latency_ms || 0} ms`;
            }
        } catch (err) {
            console.error("Prediction error:", err);
            elements.latencyStat.textContent = `Error connecting to API`;
        } finally {
            state.isPredicting = false;
            elements.btnPredict.classList.remove("loading");
            updateGhostOverlay();
        }
    }

    // Render Suggestion Chips
    function renderChips(predictions) {
        elements.chipsContainer.innerHTML = "";

        predictions.slice(0, 6).forEach((item, index) => {
            const chip = document.createElement("button");
            chip.className = `word-chip ${index === 0 ? "top-choice" : ""}`;
            chip.innerHTML = `
                <span class="chip-word">${escapeHtml(item.word)}</span>
                <span class="chip-pct">${item.percentage.toFixed(1)}%</span>
            `;

            chip.addEventListener("click", () => {
                const currentText = elements.textInput.value;
                const needsSpace = currentText.length > 0 && !currentText.endsWith(" ");
                elements.textInput.value = currentText + (needsSpace ? " " : "") + item.word + " ";
                elements.textInput.focus();
                syncStats();
                requestPrediction(true);
            });

            elements.chipsContainer.appendChild(chip);
        });
    }

    // Render Probability Bar Chart
    function renderProbabilities(predictions) {
        elements.probabilitiesList.innerHTML = "";

        predictions.forEach(item => {
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-header">
                    <div class="prob-word-wrap">
                        <span class="prob-rank">#${item.rank}</span>
                        <span class="prob-word">${escapeHtml(item.word)}</span>
                    </div>
                    <span class="prob-value">${item.percentage.toFixed(2)}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
            `;

            // Click to insert word
            row.addEventListener("click", () => {
                const currentText = elements.textInput.value;
                const needsSpace = currentText.length > 0 && !currentText.endsWith(" ");
                elements.textInput.value = currentText + (needsSpace ? " " : "") + item.word + " ";
                elements.textInput.focus();
                syncStats();
                requestPrediction(true);
            });

            elements.probabilitiesList.appendChild(row);

            // Animate bar fill
            setTimeout(() => {
                const fill = row.querySelector(".prob-bar-fill");
                if (fill) fill.style.width = `${Math.min(100, Math.max(3, item.percentage))}%`;
            }, 30);
        });
    }

    // Render Neural Token Inspector
    function renderInspector(info) {
        if (!info) return;

        elements.inspectCleaned.textContent = `"${info.cleaned_text || ""}"`;
        elements.inspectTokens.textContent = JSON.stringify(info.raw_tokens || []);
        elements.inspectTokenIds.textContent = JSON.stringify(info.token_ids || []);

        // Vector display
        elements.inspectVector.innerHTML = "";
        const seq = info.padded_sequence || [];
        if (seq.length === 0) {
            elements.inspectVector.innerHTML = `<span class="vector-empty">Vector empty</span>`;
            return;
        }

        seq.forEach(id => {
            const pill = document.createElement("span");
            pill.className = `vector-pill ${id > 0 ? "active" : ""}`;
            pill.textContent = id;
            elements.inspectVector.appendChild(pill);
        });
    }

    // Multi-Word Generation
    async function generateMultiWord() {
        const text = elements.textInput.value.trim();
        if (!text) {
            showToast("Please enter some seed text in the editor first.");
            elements.textInput.focus();
            return;
        }

        if (state.isGenerating) return;
        state.isGenerating = true;
        elements.btnGenerate.innerHTML = `
            <svg class="spinner" viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10"></path>
            </svg>
            <span>Generating...</span>
        `;

        try {
            const maxWords = parseInt(elements.wordCountSlider.value) || 5;
            const temperature = parseFloat(elements.tempSlider.value) || 0.8;
            const mode = elements.genModeSelect.value;

            const res = await fetch(`${state.apiBase}/api/generate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    max_words: maxWords,
                    temperature: temperature,
                    mode: mode
                })
            });

            const data = await res.json();
            if (data.success && data.added_words && data.added_words.length > 0) {
                elements.genResultBox.style.display = "block";
                
                // Construct styled result
                const wordsHtml = data.added_words.map(w => `<span class="gen-highlight">${escapeHtml(w)}</span>`).join(" ");
                elements.genResultText.innerHTML = `<strong>${escapeHtml(data.original_text)}</strong> ${wordsHtml}`;
                elements.genResultBox.dataset.completedText = data.completed_text;
                showToast(`Generated ${data.added_words.length} words in ${data.latency_ms} ms!`);
            } else {
                elements.genResultBox.style.display = "block";
                elements.genResultText.textContent = data.error || "No additional words could be generated.";
            }
        } catch (err) {
            console.error("Generation error:", err);
            showToast("Failed to generate words.");
        } finally {
            state.isGenerating = false;
            elements.btnGenerate.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                <span>Generate Text</span>
            `;
        }
    }

    function applyGeneratedText() {
        const completed = elements.genResultBox.dataset.completedText;
        if (completed) {
            elements.textInput.value = completed + " ";
            elements.textInput.focus();
            syncStats();
            requestPrediction(true);
            showToast("Applied generated text to editor.");
        }
    }

    // Utilities
    function clearEditor() {
        elements.textInput.value = "";
        clearPredictions();
        elements.genResultBox.style.display = "none";
        syncStats();
        elements.textInput.focus();
    }

    function clearPredictions() {
        state.currentPredictions = [];
        state.topWord = "";
        elements.chipsContainer.innerHTML = `<span class="chip-placeholder">Type a sentence to see instant word predictions...</span>`;
        elements.probabilitiesList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📊</div>
                <p>No predictions computed yet.</p>
                <small>Start typing on the left to see neural network confidence levels.</small>
            </div>
        `;
        elements.inspectCleaned.textContent = "--";
        elements.inspectTokens.textContent = "[]";
        elements.inspectTokenIds.textContent = "[]";
        elements.inspectVector.innerHTML = `<span class="vector-empty">Vector empty</span>`;
        elements.latencyStat.textContent = "Latency: -- ms";
        updateGhostOverlay();
    }

    function copyText() {
        const text = elements.textInput.value;
        if (!text) {
            showToast("Nothing to copy!");
            return;
        }

        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                showToast("Copied to clipboard!");
            });
        } else {
            elements.textInput.select();
            document.execCommand("copy");
            showToast("Copied to clipboard!");
        }
    }

    function showToast(message) {
        elements.toast.textContent = message;
        elements.toast.classList.add("show");
        setTimeout(() => {
            elements.toast.classList.remove("show");
        }, 2500);
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Launch app once DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
