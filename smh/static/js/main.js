class ChatApp {
  constructor() {
    // WebSocket 관리
    this.ws = null;
    this.currentModel = "meta";
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;

    // UI 요소 초기화
    this.initializeElements();

    // 상태 관리
    this.uploadedFiles = new Map();
    this.isProcessing = false;

    // 이벤트 리스너 설정
    this.setupEventListeners();

    // WebSocket 연결 시작
    this.connectWebSocket();
  }

  initializeElements() {
    this.elements = {
      connectionIndicator: document.getElementById("connection-indicator"),
      connectionText: document.getElementById("connection-text"),
      modelButtons: document.querySelectorAll(".model-button"),
      chatMessages: document.getElementById("chat-messages"),
      messageInput: document.getElementById("message-input"),
      sendButton: document.getElementById("send-button"),
      fileInput: document.getElementById("fileInput"),
      dropArea: document.getElementById("dropArea"),
      dropOverlay: document.getElementById("dropOverlay"),
      uploadedFiles: document.getElementById("uploadedFiles"),
      uploadProgress: document.getElementById("upload-progress"),
    };
  }

  setupEventListeners() {
    // 모델 선택 버튼 이벤트
    this.elements.modelButtons.forEach((button) => {
      button.addEventListener("click", () => this.handleModelChange(button));
    });

    // 메시지 전송 이벤트
    this.elements.sendButton.addEventListener("click", () =>
      this.sendMessage()
    );
    this.elements.messageInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // 파일 업로드 이벤트
    this.setupFileUploadListeners();
  }

  setupFileUploadListeners() {
    const dropArea = this.elements.dropArea;
    const dropOverlay = this.elements.dropOverlay;

    // Drag & Drop 이벤트
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      dropArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    dropArea.addEventListener("dragenter", () =>
      dropOverlay.classList.add("active")
    );
    dropArea.addEventListener("dragleave", () =>
      dropOverlay.classList.remove("active")
    );
    dropArea.addEventListener("drop", (e) => {
      dropOverlay.classList.remove("active");
      const files = Array.from(e.dataTransfer.files);
      this.handleFiles(files);
    });

    // 파일 입력 이벤트
    this.elements.fileInput.addEventListener("change", (e) => {
      const files = Array.from(e.target.files);
      this.handleFiles(files);
    });
  }

  async handleFiles(files) {
    for (const file of files) {
      if (!this.validateFile(file)) continue;

      try {
        this.showUploadProgress(true);
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/mainupload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error("Upload failed");

        const result = await response.json();
        this.addUploadedFile(file.name, result);
        this.showToast("success", "파일이 성공적으로 업로드되었습니다.");
      } catch (error) {
        this.showToast("error", "파일 업로드 중 오류가 발생했습니다.");
        console.error("File upload error:", error);
      } finally {
        this.showUploadProgress(false);
      }
    }
  }

  validateFile(file) {
    const allowedTypes = [".pdf", ".hwp", ".hwpx", ".doc", ".docx"];
    const extension = "." + file.name.split(".").pop().toLowerCase();

    if (!allowedTypes.includes(extension)) {
      this.showToast("error", "지원하지 않는 파일 형식입니다.");
      return false;
    }

    if (file.size > 10 * 1024 * 1024) {
      // 10MB limit
      this.showToast("error", "파일 크기는 10MB를 초과할 수 없습니다.");
      return false;
    }

    return true;
  }

  addUploadedFile(filename, fileData) {
    const fileElement = document.createElement("div");
    fileElement.className = "uploaded-file";
    fileElement.innerHTML = `
          <div class="file-info">
              <i class="fas fa-file-alt"></i>
              <span>${filename}</span>
          </div>
          <button class="remove-file">
              <i class="fas fa-times"></i>
          </button>
      `;

    fileElement.querySelector(".remove-file").addEventListener("click", () => {
      this.uploadedFiles.delete(filename);
      fileElement.remove();
    });

    this.elements.uploadedFiles.appendChild(fileElement);
    this.uploadedFiles.set(filename, fileData);
  }

  connectWebSocket() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(`ws://${window.location.host}/chat`);

    this.ws.onopen = () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.updateConnectionStatus(true);
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      this.updateConnectionStatus(false);
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.showToast("error", "WebSocket 연결 오류가 발생했습니다.");
    };

    this.ws.onmessage = (event) => this.handleWebSocketMessage(event);
  }

  async sendMessage() {
    const content = this.elements.messageInput.value.trim();
    if (!content || this.isProcessing) return;

    try {
      this.isProcessing = true;
      this.elements.messageInput.disabled = true;
      this.elements.sendButton.disabled = true;

      // 메시지 추가 및 UI 업데이트
      this.addMessage("user", content);
      this.elements.messageInput.value = "";

      // WebSocket으로 메시지 전송
      const message = {
        type: "message",
        content: content,
        model: this.currentModel,
        files: Array.from(this.uploadedFiles.values()),
      };

      this.ws.send(JSON.stringify(message));
    } catch (error) {
      console.error("Send message error:", error);
      this.showToast("error", "메시지 전송 중 오류가 발생했습니다.");
    } finally {
      this.isProcessing = false;
      this.elements.messageInput.disabled = false;
      this.elements.sendButton.disabled = false;
    }
  }

  handleWebSocketMessage(event) {
    try {
      console.log("WebSocket message received:", event.data);
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "assistant":
          if (data.streaming) {
            this.updateStreamingMessage(data.content, data.model);
          } else {
            // non-streaming 응답 처리 추가
            this.addMessage("assistant", data.content, data.model);
          }
          if (data.isFullResponse) {
            this.completeStreamingMessage();
          }
          break;

        case "error":
          this.showToast("error", data.content);
          break;

        default:
          console.warn("Unknown message type:", data.type);
      }
    } catch (error) {
      console.error("Message handling error:", error);
    }
  }

  handleModelChange(button) {
    const model = button.dataset.model;
    if (model === this.currentModel) return;

    // UI 업데이트
    this.elements.modelButtons.forEach((btn) => {
      btn.classList.toggle("active", btn === button);
    });

    this.currentModel = model;
    this.showToast("info", `${model.toUpperCase()} 모델로 전환되었습니다.`);
  }

  addMessage(role, content, model = null) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;
    if (model) messageDiv.classList.add(`${model}-content`);

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";

    if (role === "assistant") {
      const img = document.createElement("img");
      img.src = `/static/image/${model || this.currentModel}.png`;
      img.alt = `${model || this.currentModel} Avatar`;
      avatar.appendChild(img);
    } else {
      avatar.innerHTML = '<i class="fas fa-user"></i>';
    }

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";

    // 마크다운 처리 (Gemini 모델인 경우)
    if (
      role === "assistant" &&
      (model === "gemini" || this.currentModel === "gemini")
    ) {
      messageContent.innerHTML = this.formatMarkdown(content);
    } else {
      messageContent.textContent = content;
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    this.elements.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();

    return messageDiv;
  }

  updateStreamingMessage(content, model) {
    console.log("updateStreamingMessage 호출됨:", model, content);

    let messageDiv = this.elements.chatMessages.querySelector(
      ".message.assistant:last-child"
    );
    if (!messageDiv) {
      console.log("새 메시지 생성");
      messageDiv = this.addMessage("assistant", "", model);
    }

    const messageContent = messageDiv.querySelector(".message-content");

    // 원본 텍스트를 데이터 속성에 저장
    if (!messageContent.dataset.originalText) {
      messageContent.dataset.originalText = "";
    }

    // 원본 텍스트에 새 컨텐츠 추가
    messageContent.dataset.originalText += content;

    console.log("누적된 원본 텍스트:", messageContent.dataset.originalText);

    // 모델이 Gemini인 경우 마크다운 처리
    if (model === "gemini") {
      console.log("Gemini 메시지 마크다운 처리 시작");
      try {
        messageContent.innerHTML = this.formatMarkdown(
          messageContent.dataset.originalText
        );
        console.log("마크다운 처리 완료");
      } catch (error) {
        console.error("마크다운 처리 중 오류:", error);
        messageContent.textContent = messageContent.dataset.originalText;
      }
    } else {
      messageContent.textContent = messageContent.dataset.originalText;
    }

    this.scrollToBottom();
  }

  // 마크다운 포맷팅 함수 추가
  formatMarkdown(text) {
    console.log("formatMarkdown 함수 호출됨");

    if (!text) {
      console.log("텍스트가 비어있음");
      return "";
    }

    console.log("마크다운 변환 전:", text);

    if (typeof marked === "undefined") {
      console.error("marked가 정의되지 않았습니다!");
      return text;
    }

    try {
      const result = marked.parse(text, {
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false,
      });
      console.log("마크다운 변환 후:", result);
      return result;
    } catch (error) {
      console.error("마크다운 변환 중 오류:", error);
      return text;
    }
  }

  scrollToBottom() {
    this.elements.chatMessages.scrollTop =
      this.elements.chatMessages.scrollHeight;
  }

  updateConnectionStatus(connected) {
    const indicator = this.elements.connectionIndicator;
    const text = this.elements.connectionText;

    if (connected) {
      indicator.classList.remove("offline");
      indicator.classList.add("online");
      text.textContent = "연결됨";
    } else {
      indicator.classList.remove("online");
      indicator.classList.add("offline");
      text.textContent = "연결 끊김";
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.showToast(
        "error",
        "연결을 재시도할 수 없습니다. 페이지를 새로고침해주세요."
      );
      return;
    }

    this.reconnectAttempts++;
    setTimeout(() => this.connectWebSocket(), 2000 * this.reconnectAttempts);
  }

  showToast(type, message) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;

    const container = document.getElementById("toast-container");
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("fade-out");
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  showUploadProgress(show) {
    this.elements.uploadProgress.classList.toggle("hidden", !show);
  }
}

// 앱 초기화
document.addEventListener("DOMContentLoaded", () => {
  // marked.js 로드 확인
  console.log("DOMContentLoaded 이벤트 발생");
  if (typeof marked === "undefined") {
    console.error("marked.js가 로드되지 않았습니다!");
  } else {
    console.log("marked.js 로드됨, 테스트:", marked.parse("**테스트**"));
  }

  window.chatApp = new ChatApp();
});
