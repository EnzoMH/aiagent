const generateUUID = () =>
  "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c == "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });

// 상태 관리를 위한 상수 정의는 그대로 유지
const ProcessState = {
  IDLE: "idle",
  FILE_SELECTED: "file_selected",
  ANALYZING: "analyzing",
  ANALYSIS_COMPLETE: "analysis_complete",
  GENERATING_TOC: "generating_toc",
  TOC_COMPLETE: "toc_complete",
  GENERATING_CONTENT: "generating_content",
  COMPLETED: "completed",
  ERROR: "error",
};

// 응답 데이터 타입 정의를 더 유연하게 변경
const ResponseTypes = {
  // 공통 응답 구조
  COMMON: {
    status: "string",
    message: "string",
    errors: "array?",
    timestamp: "string",
  },

  // 업로드 응답
  UPLOAD: {
    required: {
      file_id: "string",
      filename: "string",
    },
    optional: {
      file_type: "string",
      page_count: "number",
      metadata: {
        text: "string",
        table_sections: "array",
        initial_analysis: {
          keywords: "array",
          page_count: "number",
          // 추가 필드들은 옵셔널
          budget: "object?",
          duration: "object?",
          sections: "array?",
        },
      },
    },
  },

  // 분석 응답
  ANALYSIS: {
    required: {
      analysis_id: "string",
      created_at: "string",
    },
    optional: {
      data: {
        sections: "object",
        metadata: {
          budget: {
            amount: "number?",
            unit: "string?",
            raw: "string?",
          },
          duration: {
            start_date: "string?",
            end_date: "string?",
            months: "number?",
            raw: "string?",
          },
          page_limit: "number?",
          presentation_time: "number?",
          requirements: "object?",
        },
        evaluation: {
          descriptive: {
            total: "number?",
            items: "array?",
            scores: "object?",
          },
          price: {
            total: "number?",
            items: "array?",
            scores: "object?",
          },
        },
      },
    },
  },

  // TOC 응답
  TOC: {
    required: {
      toc_id: "string",
      created_at: "string",
      "data.structure": "array", // 중첩 경로 표현
    },
    optional: {
      data: {
        page_allocations: "object",
        total_pages: "number",
        presentation_time: "number",
        metadata: {
          generated_at: "string?",
          section_count: "number?",
        },
      },
    },
  },
};

// 상태 매핑에 fallback 상태 추가
const StatusMapping = {
  success: {
    upload: ProcessState.FILE_SELECTED,
    analysis: ProcessState.ANALYSIS_COMPLETE,
    toc: ProcessState.TOC_COMPLETE,
    generate: ProcessState.COMPLETED,
    default: ProcessState.COMPLETED, // 기본값 추가
  },
  error: ProcessState.ERROR,
  pending: ProcessState.IDLE,
  processing: {
    analysis: ProcessState.ANALYZING,
    toc: ProcessState.GENERATING_TOC,
    generate: ProcessState.GENERATING_CONTENT,
    default: ProcessState.ANALYZING, // 기본값 추가
  },
  // fallback 상태 추가
  fallback: {
    upload: ProcessState.FILE_SELECTED,
    analysis: ProcessState.ANALYSIS_COMPLETE,
    toc: ProcessState.TOC_COMPLETE,
    generate: ProcessState.COMPLETED,
  },
};

// Logger 클래스를 stateManager에 독립적으로 만듭니다
const Logger = {
  levels: {
    INFO: "info",
    WARN: "warn",
    ERROR: "error",
    DEBUG: "debug",
  },

  log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMessage = {
      timestamp,
      level,
      message,
      data,
    };

    console.log(
      `[${timestamp}] [${level.toUpperCase()}] ${message}`,
      data ? data : ""
    );

    // UI에 알림 표시 (에러와 경고의 경우)
    if (level === this.levels.ERROR || level === this.levels.WARN) {
      this.showNotification(message, level);
    }

    this.saveLog(logMessage);
  },

  showNotification(message, type = "info") {
    const notifications = document.getElementById("notifications");
    if (!notifications) return;

    const notification = document.createElement("div");
    notification.className = `p-4 mb-4 rounded-lg ${
      type === "error" ? "bg-red-100 text-red-700" : "bg-blue-100 text-blue-700"
    }`;
    notification.textContent = message;

    // 기존 알림 제거
    while (notifications.firstChild) {
      notifications.removeChild(notifications.firstChild);
    }

    // 새 알림 추가
    notifications.appendChild(notification);

    // 3초 후 자동 제거
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, 3000);
  },

  info(message, data = null) {
    this.log(this.levels.INFO, message, data);
  },

  warn(message, data = null) {
    this.log(this.levels.WARN, message, data);
  },

  error(message, data = null) {
    this.log(this.levels.ERROR, message, data);
  },

  debug(message, data = null) {
    this.log(this.levels.DEBUG, message, data);
  },

  saveLog(logMessage) {
    // 로컬 스토리지에 로그 저장
    const logs = JSON.parse(localStorage.getItem("app_logs") || "[]");
    logs.push(logMessage);
    localStorage.setItem("app_logs", JSON.stringify(logs.slice(-100))); // 최근 100개 로그만 유지
  },
};

// API 응답 검증 유틸리티
class ResponseValidator {
  static ERROR_MESSAGES = {
    UPLOAD: {
      VALIDATION: "업로드 응답 검증 실패",
      METADATA: "메타데이터 검증 실패",
      ERROR: "업로드 응답 검증 중 오류 발생",
    },
    ANALYSIS: {
      VALIDATION: "분석 응답 검증 실패",
      ERROR: "분석 응답 검증 중 오류 발생",
    },
    TOC: {
      VALIDATION: "TOC 응답 검증 실패",
      ERROR: "TOC 응답 검증 중 오류 발생",
    },
  };

  // 필수/선택 필드 정의를 명확히 분리
  static REQUIRED_FIELDS = {
    UPLOAD: ["file_id", "filename"],
    ANALYSIS: ["analysis_id"],
    TOC: ["toc_id", "structure"],
  };

  static validateResponse(data, type) {
    try {
      // 기본 응답 구조 검증
      if (!data || typeof data !== "object") {
        Logger.warn(`${type} 응답이 객체가 아님`, data);
        return { isValid: false, warning: true };
      }

      // 필수 필드만 엄격하게 검증
      const requiredFields = this.REQUIRED_FIELDS[type];
      const missingFields = requiredFields.filter(
        (field) => !this._getNestedValue(data, field)
      );

      if (missingFields.length > 0) {
        Logger.warn(
          `${type} 필수 필드 누락: ${missingFields.join(", ")}`,
          data
        );
        return { isValid: false, warning: true, missingFields };
      }

      // 나머지 필드는 유연하게 처리
      return { isValid: true };
    } catch (error) {
      Logger.error(`${type} 응답 검증 중 오류`, error);
      return { isValid: false, error };
    }
  }

  // 중첩된 객체에서 값을 안전하게 가져오는 유틸리티 메서드
  static _getNestedValue(obj, path) {
    return path.split(".").reduce((current, part) => {
      return current && current[part] !== undefined ? current[part] : undefined;
    }, obj);
  }

  // 타입 검증을 유연하게 처리하는 메서드
  static _validateType(value, expectedType) {
    if (value === undefined || value === null) {
      return true; // 선택적 필드는 누락 허용
    }

    switch (expectedType) {
      case "string":
        return typeof value === "string" || value instanceof String;
      case "number":
        return !isNaN(Number(value)); // 문자열로 된 숫자도 허용
      case "array":
        return Array.isArray(value);
      case "object":
        return typeof value === "object" && !Array.isArray(value);
      default:
        return true; // 알 수 없는 타입은 허용
    }
  }

  // 각 엔드포인트별 커스텀 검증 메서드
  static validateUploadResponse(data) {
    const result = this.validateResponse(data, "UPLOAD");
    if (!result.isValid && !result.warning) {
      return false;
    }

    // 메타데이터는 유연하게 처리
    if (data.metadata) {
      const metadata = data.metadata;
      if (typeof metadata !== "object") {
        Logger.warn("메타데이터가 객체가 아님, 기본값 사용", metadata);
        data.metadata = {}; // 기본값 제공
      }
    }

    return true;
  }

  static validateAnalysisResponse(data) {
    const result = this.validateResponse(data, "ANALYSIS");
    if (!result.isValid && !result.warning) {
      return false;
    }

    // 분석 데이터 유연하게 처리
    if (data.data) {
      // sections, metadata, evaluation이 없으면 기본값 제공
      data.data.sections = data.data.sections || {};
      data.data.metadata = data.data.metadata || {};
      data.data.evaluation = data.data.evaluation || {
        descriptive: { total: 80, items: [] },
        price: { total: 20, items: [] },
      };
    }

    return true;
  }

  static validateTocResponse(data) {
    const result = this.validateResponse(data, "TOC");
    if (!result.isValid && !result.warning) {
      return false;
    }

    // TOC 데이터 유연하게 처리
    if (data.data) {
      data.data.page_allocations = data.data.page_allocations || {};
      data.data.metadata = data.data.metadata || {};
    }

    return true;
  }
}

class APIService {
  constructor() {
    this.baseURL = "/api";
  }

  async uploadFile(file) {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${this.baseURL}/upload`, {
        method: "POST",
        body: formData,
        headers: {
          // Content-Type은 FormData를 사용할 때는 설정하지 않습니다
          Accept: "application/json",
        },
        credentials: "same-origin", // 쿠키 포함
      });

      if (!response.ok) {
        throw new Error(`Upload failed with status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      Logger.error("File upload failed", error);
      throw error;
    }
  }

  async analyzeDocument(fileId, options = {}) {
    try {
      const response = await fetch(`${this.baseURL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        credentials: "same-origin",
        body: JSON.stringify({
          file_id: fileId,
          options: options,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.message || "RFP 문서 분석 실패");
      }

      const data = await response.json();
      return data;
    } catch (error) {
      Logger.error("문서 분석 실패", error);
      throw error;
    }
  }

  async generateTOC(analysisId, params) {
    try {
      const response = await fetch(`${this.baseURL}/toc`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_id: analysisId,
          ...params,
          min_pages_per_section: params.min_pages_per_section || 2,
        }),
      });

      if (!response.ok) {
        throw new Error("TOC generation failed");
      }

      const data = await response.json();
      return this._enrichResponse(data, "TOC");
    } catch (error) {
      Logger.error("TOC generation failed", error);
      throw error;
    }
  }

  async *generateContent(sectionId, tocId, stylePreferences = {}) {
    try {
      const response = await fetch(`${this.baseURL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          section_id: sectionId,
          toc_id: tocId,
          style_preferences: stylePreferences,
          max_retries: 3,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("Generate request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const event = JSON.parse(decoder.decode(value));
        // 기본 이벤트 구조 제공
        const enrichedEvent = {
          event_type: event.event_type || "progress",
          section_id: event.section_id || sectionId,
          progress: event.progress || 0,
          content: event.content || null,
          message: event.message || null,
          timestamp: event.timestamp || new Date().toISOString(),
        };

        yield enrichedEvent;
      }
    } catch (error) {
      Logger.error("Content generation failed", error);
      throw error;
    }
  }
}

class ModalController {
  constructor() {
    this.modal = document.getElementById("modal");
    this.title = document.getElementById("modal-title");
    this.content = document.getElementById("modal-content");
    this.actions = document.getElementById("modal-actions");
    this.closeBtn = document.getElementById("modal-close");

    // 닫기 버튼 이벤트 리스너
    this.closeBtn.addEventListener("click", () => this.hide());

    // 모달 외부 클릭 시 닫기
    this.modal.addEventListener("click", (e) => {
      if (e.target === this.modal) this.hide();
    });
  }

  show(title, content, actions = []) {
    this.title.textContent = title;

    // content가 HTML 문자열이나 DOM 엘리먼트일 수 있음
    if (typeof content === "string") {
      this.content.innerHTML = content;
    } else if (content instanceof Element) {
      this.content.innerHTML = "";
      this.content.appendChild(content);
    }

    // 액션 버튼 초기화
    this.actions.innerHTML = "";

    // 액션 버튼 추가
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.textContent = action.text;
      button.className = `px-4 py-2 rounded ${action.class || "bg-gray-200"}`;
      button.onclick = action.onClick;
      this.actions.appendChild(button);
    });

    this.modal.classList.remove("hidden");
    this.modal.classList.add("flex");
  }

  hide() {
    this.modal.classList.remove("flex");
    this.modal.classList.add("hidden");
  }

  // 에러 표시를 위한 유틸리티 메서드
  showError(title, message, retryAction = null) {
    const actions = [
      {
        text: "확인",
        class: "bg-gray-200 hover:bg-gray-300",
        onClick: () => this.hide(),
      },
    ];

    if (retryAction) {
      actions.unshift({
        text: "재시도",
        class: "bg-blue-500 hover:bg-blue-600 text-white",
        onClick: retryAction,
      });
    }

    this.show(title, `<div class="text-red-600">${message}</div>`, actions);
  }

  // 진행 상태 표시를 위한 유틸리티 메서드
  showProgress(title, message) {
    this.show(
      title,
      `<div class="space-y-4">
        <p>${message}</p>
        <div class="w-full h-2 bg-gray-200 rounded-full">
          <div class="progress-bar h-full bg-blue-600 rounded-full" style="width: 0%"></div>
        </div>
      </div>`
    );
  }

  // 진행률 업데이트
  updateProgress(percent) {
    const progressBar = this.content.querySelector(".progress-bar");
    if (progressBar) {
      progressBar.style.width = `${percent}%`;
    }
  }
}

class UIStateManager {
  constructor(elements) {
    this.elements = elements;
  }

  // updateSections 메서드 추가
  updateSections(state) {
    const { analysisSection, tocSection, generationSection } = this.elements;

    // 각 섹션의 표시 여부 결정
    const sectionVisibility = {
      analysis: [
        ProcessState.ANALYZING,
        ProcessState.ANALYSIS_COMPLETE,
      ].includes(state),
      toc: [ProcessState.GENERATING_TOC, ProcessState.TOC_COMPLETE].includes(
        state
      ),
      generation: [
        ProcessState.GENERATING_CONTENT,
        ProcessState.COMPLETED,
      ].includes(state),
    };

    // 섹션 표시/숨김 처리
    if (analysisSection) {
      analysisSection.classList.toggle("hidden", !sectionVisibility.analysis);
    }
    if (tocSection) {
      tocSection.classList.toggle("hidden", !sectionVisibility.toc);
    }
    if (generationSection) {
      generationSection.classList.toggle(
        "hidden",
        !sectionVisibility.generation
      );
    }
  }

  // 버튼 상태 업데이트 메서드 추가
  updateButtons(state) {
    // 모든 버튼 초기화
    const buttons = {
      "btn-analyze": this.elements.analyzeBtn,
      "btn-generate-toc": this.elements.generateTocBtn,
      "btn-generate-content": this.elements.generateContentBtn,
      "btn-download": this.elements.downloadBtn,
    };

    // 모든 버튼 숨기고 비활성화
    Object.values(buttons).forEach((button) => {
      if (button) {
        button.classList.add("hidden");
        button.disabled = true;
      }
    });

    // 상태에 따른 버튼 표시 및 활성화
    switch (state) {
      case ProcessState.FILE_SELECTED:
        if (buttons["btn-analyze"]) {
          buttons["btn-analyze"].classList.remove("hidden");
          buttons["btn-analyze"].disabled = false;
        }
        break;

      case ProcessState.ANALYSIS_COMPLETE:
        if (buttons["btn-generate-toc"]) {
          buttons["btn-generate-toc"].classList.remove("hidden");
          buttons["btn-generate-toc"].disabled = false;
        }
        break;

      case ProcessState.TOC_COMPLETE:
        if (buttons["btn-generate-content"]) {
          buttons["btn-generate-content"].classList.remove("hidden");
          buttons["btn-generate-content"].disabled = false;
        }
        break;

      case ProcessState.COMPLETED:
        if (buttons["btn-download"]) {
          buttons["btn-download"].classList.remove("hidden");
          buttons["btn-download"].disabled = false;
        }
        break;
    }
  }

  // 기존 메서드들...
  updateStatusBadge(state) {
    if (this.elements.statusBadge) {
      const badge = this.elements.statusBadge;
      badge.classList.remove("hidden");

      // 상태에 따른 배지 스타일 설정
      const stateStyles = {
        [ProcessState.FILE_SELECTED]: "bg-blue-100 text-blue-800",
        [ProcessState.ANALYZING]: "bg-yellow-100 text-yellow-800",
        [ProcessState.ANALYSIS_COMPLETE]: "bg-green-100 text-green-800",
        [ProcessState.GENERATING_TOC]: "bg-yellow-100 text-yellow-800",
        [ProcessState.TOC_COMPLETE]: "bg-green-100 text-green-800",
        [ProcessState.GENERATING_CONTENT]: "bg-yellow-100 text-yellow-800",
        [ProcessState.COMPLETED]: "bg-green-100 text-green-800",
        [ProcessState.ERROR]: "bg-red-100 text-red-800",
      };

      // 모든 스타일 클래스 제거
      Object.values(stateStyles).forEach((style) => {
        const classes = style.split(" ");
        badge.classList.remove(...classes);
      });

      // 현재 상태에 맞는 스타일 적용
      const currentStyle = stateStyles[state];
      if (currentStyle) {
        const classes = currentStyle.split(" ");
        badge.classList.add(...classes);
      }

      // 상태 텍스트 설정
      badge.textContent = state;
    }
  }

  updateUI() {
    Logger.debug("Updating UI", {
      currentState: this.currentState,
      visibleButtons: this.elements
        ? Object.entries(this.elements)
            .filter(
              ([key, el]) =>
                key.startsWith("btn-") && el && !el.classList.contains("hidden")
            )
            .map(([key]) => key)
        : [],
    });

    // UI 매니저를 통해 업데이트
    this.updateStatusBadge(this.currentState);
    this.updateButtons(this.currentState);
    this.updateSections(this.currentState);
  }
}

class EventHandlers {
  constructor(stateManager, modalController, elements) {
    this.stateManager = stateManager;
    this.modalController = modalController;
    this.elements = elements;
    this.apiService = stateManager.apiService;
    this.retryCount = new Map();
    this.MAX_RETRIES = 3;

    // 바인딩은 유지
    this.handleFileUpload = this.handleFileUpload.bind(this);
    this.handleAnalysisProgress = this.handleAnalysisProgress.bind(this);
    this.handleTocGeneration = this.handleTocGeneration.bind(this);
    this.handleContentGeneration = this.handleContentGeneration.bind(this);
  }

  // 파일 업로드 핸들러
  async handleFileUpload(file) {
    try {
      if (!this.validateFile(file)) return;

      this.modalController.showProgress(
        "파일 업로드",
        "파일을 업로드하고 있습니다..."
      );
      const apiService = this.apiService;
      const uploadResult = await apiService.uploadFile(file);

      // UI 업데이트
      this.elements.fileName.textContent = file.name;
      this.elements.fileInfo.classList.remove("hidden");
      this.elements.dropZone.classList.add("hidden");

      // 상태 업데이트
      this.stateManager.setState(ProcessState.FILE_SELECTED, uploadResult);
      this.modalController.hide();

      Logger.info("파일 업로드 완료", { filename: file.name });
    } catch (error) {
      await this.handleError("upload", error, () =>
        this.handleFileUpload(file)
      );
    }
  }

  // 분석 진행 상태 핸들러
  async handleAnalysisProgress(fileId) {
    try {
      Logger.info("문서 분석 시작", { fileId });
      this.stateManager.setState(ProcessState.ANALYZING);

      const analysisResult = await this.apiService.analyzeDocument(fileId);

      if (!analysisResult || !analysisResult.data) {
        throw new Error("분석 결과가 유효하지 않습니다");
      }

      // 분석 결과 UI 업데이트
      this.updateAnalysisSection(analysisResult.data);

      this.stateManager.setState(
        ProcessState.ANALYSIS_COMPLETE,
        analysisResult
      );
      this.showNotification("문서 분석이 완료되었습니다.", "info");
    } catch (error) {
      Logger.error("analysis 작업 실패", error);
      this.stateManager.setState(ProcessState.ERROR, { error });
      this.showNotification("문서 분석에 실패했습니다.", "error");
    }
  }

  // 목차 생성 핸들러
  async handleTocGeneration(analysisId, params) {
    try {
      const apiService = this.apiService;
      this.updateProgressUI("목차 생성", 0);
      const result = await apiService.generateTOC(analysisId, params);

      // 목차 UI 업데이트
      this.updateTocSection(result.data);
      this.stateManager.setState(ProcessState.TOC_COMPLETE, result);

      Logger.info("목차 생성 완료", { tocId: result.toc_id });
    } catch (error) {
      await this.handleError("toc", error, () =>
        this.handleTocGeneration(analysisId, params)
      );
    }
  }

  // 컨텐츠 생성 핸들러
  async handleContentGeneration(sectionId, tocId) {
    try {
      const apiService = this.apiService;
      this.updateProgressUI("컨텐츠 생성", 0);
      let accumulatedContent = {};

      const generator = apiService.generateContent(sectionId, tocId);
      for await (const event of generator) {
        switch (event.event_type) {
          case "progress":
            this.updateProgressUI("컨텐츠 생성", event.progress);
            if (event.content) {
              accumulatedContent = { ...accumulatedContent, ...event.content };
              this.updateContentPreview(accumulatedContent);
            }
            break;

          case "complete":
            this.stateManager.setState(ProcessState.COMPLETED, {
              section_id: sectionId,
              content: event.content || accumulatedContent,
            });
            break;

          case "error":
            throw new Error(event.message);
        }
      }

      Logger.info("컨텐츠 생성 완료", { sectionId });
    } catch (error) {
      await this.handleError("content", error, () =>
        this.handleContentGeneration(sectionId, tocId)
      );
    }
  }

  // 유틸리티 메서드들
  validateFile(file) {
    if (!file) {
      this.modalController.showError(
        "파일 오류",
        "파일이 선택되지 않았습니다."
      );
      return false;
    }

    const validTypes = [".pdf", ".hwp", ".hwpx"];
    const fileExtension = "." + file.name.split(".").pop().toLowerCase();

    if (!validTypes.includes(fileExtension)) {
      this.modalController.showError(
        "파일 형식 오류",
        `지원하지 않는 파일 형식입니다. (지원 형식: ${validTypes.join(", ")})`
      );
      return false;
    }

    return true;
  }

  async handleError(operation, error, retryCallback) {
    const retryCount = this.retryCount.get(operation) || 0;

    if (retryCount < this.MAX_RETRIES) {
      this.retryCount.set(operation, retryCount + 1);
      this.modalController.showError(
        "오류 발생",
        `${error.message}\n재시도 중... (${retryCount + 1}/${
          this.MAX_RETRIES
        })`,
        retryCallback
      );
    } else {
      this.retryCount.delete(operation);
      this.modalController.showError(
        "오류 발생",
        "최대 재시도 횟수를 초과했습니다."
      );
      this.stateManager.setState(ProcessState.ERROR);
    }

    Logger.error(`${operation} 작업 실패`, error);
  }

  updateProgressUI(operation, progress) {
    const progressBar = this.elements.progressBar;
    const progressFill = this.elements.progressFill;

    progressBar.classList.remove("hidden");
    progressFill.style.width = `${progress}%`;

    // 상태 뱃지 업데이트
    const statusBadge = this.elements.statusBadge;
    statusBadge.textContent = `${operation} ${progress}%`;
  }

  updateContentPreview(content) {
    const container = this.elements.generatedSections;
    if (!container) return;

    const contentHtml = Object.entries(content)
      .map(
        ([type, data]) => `
        <div class="mb-6">
          <h3 class="font-medium mb-2">${type}</h3>
          <div class="prose max-w-none">
            ${data}
          </div>
        </div>
      `
      )
      .join("");

    container.innerHTML = contentHtml;
  }

  updateAnalysisSection(analysisData) {
    // 분석 섹션 표시
    const analysisSection = document.getElementById("analysis-section");
    if (analysisSection) {
      analysisSection.classList.remove("hidden");
    }

    // 메타데이터 업데이트
    this.updateMetadataSection(analysisData.metadata);

    // 평가 정보 업데이트
    this.updateEvaluationSection(analysisData.evaluation);

    // 섹션 정보 업데이트
    this.updateSectionsSection(analysisData.sections);
  }

  updateMetadataSection(metadata) {
    const metadataContent = document.getElementById("metadata-content");
    if (!metadataContent) return;

    const metadataHtml = `
      <div class="bg-gray-50 p-3 rounded">
        <span class="font-medium">예산:</span> 
        <span>${metadata.budget?.amount?.toLocaleString() || "미정"} ${
      metadata.budget?.unit || "원"
    }</span>
      </div>
      <div class="bg-gray-50 p-3 rounded">
        <span class="font-medium">기간:</span> 
        <span>${metadata.duration?.months || "미정"} 개월</span>
      </div>
      <div class="bg-gray-50 p-3 rounded">
        <span class="font-medium">페이지 제한:</span> 
        <span>${metadata.page_limit || "미정"} 페이지</span>
      </div>
      <div class="bg-gray-50 p-3 rounded">
        <span class="font-medium">발표 시간:</span> 
        <span>${metadata.presentation_time || "15"} 분</span>
      </div>
    `;

    metadataContent.innerHTML = metadataHtml;
  }

  updateEvaluationSection(evaluation) {
    const evaluationContent = document.getElementById("evaluation-content");
    if (!evaluationContent) return;

    const evaluationHtml = `
      <div class="bg-gray-50 p-4 rounded">
        <div class="flex justify-between mb-2">
          <span class="font-medium">기술평가</span>
          <span class="text-blue-600">${evaluation.descriptive.total}점</span>
        </div>
      <div class="space-y-2">
          ${evaluation.descriptive.items
            .map(
              (item) => `
            <div class="flex justify-between text-sm">
              <span>${item.name}</span>
              <span>${item.score}점</span>
      </div>
          `
            )
            .join("")}
        </div>
      </div>
      <div class="bg-gray-50 p-4 rounded">
        <div class="flex justify-between mb-2">
          <span class="font-medium">가격평가</span>
          <span class="text-blue-600">${evaluation.price.total}점</span>
        </div>
        <div class="space-y-2">
          ${evaluation.price.items
            .map(
              (item) => `
            <div class="flex justify-between text-sm">
              <span>${item.name}</span>
              <span>${item.score}점</span>
        </div>
      `
            )
            .join("")}
        </div>
      </div>
    `;

    evaluationContent.innerHTML = evaluationHtml;
  }

  updateSectionsSection(sections) {
    const sectionsContent = document.getElementById("sections-content");
    if (!sectionsContent) return;

    const sectionsHtml = Object.entries(sections)
      .map(
        ([id, section]) => `
      <div class="bg-gray-50 p-4 rounded">
        <div class="flex justify-between items-start mb-2">
          <h4 class="font-medium">${section.title}</h4>
          <span class="text-sm text-gray-500">${section.type}</span>
        </div>
        <p class="text-sm text-gray-600 line-clamp-3">${section.content}</p>
        </div>
      `
      )
      .join("");

    sectionsContent.innerHTML = sectionsHtml;
  }

  updateTocSection(tocData) {
    const tocSection = document.getElementById("toc-section");
    const tocContent = document.getElementById("toc-content");
    if (!tocSection || !tocContent) return;

    tocSection.classList.remove("hidden");

    const tocHtml = tocData.structure
      .map(
        (section) => `
      <div class="bg-gray-50 p-4 rounded">
        <div class="flex justify-between items-center mb-2">
          <h4 class="font-medium">${section.title}</h4>
            <span class="text-sm text-gray-500">${
              tocData.page_allocations[section.title] || 0
            } 페이지</span>
          </div>
          ${
            section.subsections
              ? this.renderSubsections(
                  section.subsections,
                  tocData.page_allocations
                )
              : ""
          }
        </div>
      `
      )
      .join("");

    tocContent.innerHTML = tocHtml;
  }

  renderSubsections(subsections, pageAllocations, level = 1) {
    return `
      <div class="ml-${level * 4} mt-2 space-y-2">
        ${subsections
          .map(
            (sub) => `
          <div class="flex justify-between items-center p-2 bg-gray-100 rounded">
            <span class="text-sm">${sub.title}</span>
              <span class="text-xs text-gray-500">${
                pageAllocations[sub.title] || 0
              } 페이지</span>
            </div>
            ${
              sub.subsections
                ? this.renderSubsections(
                    sub.subsections,
                    pageAllocations,
                    level + 1
                  )
                : ""
            }
          `
          )
          .join("")}
      </div>
    `;
  }

  // showNotification 메서드 추가
  showNotification(message, type = "info") {
    const notifications = document.getElementById("notifications");
    if (!notifications) return;

    const notification = document.createElement("div");
    notification.className = `p-4 mb-4 rounded-lg ${
      type === "error" ? "bg-red-100 text-red-700" : "bg-blue-100 text-blue-700"
    }`;
    notification.textContent = message;

    // 기존 알림 제거
    while (notifications.firstChild) {
      notifications.removeChild(notifications.firstChild);
    }

    // 새 알림 추가
    notifications.appendChild(notification);

    // 3초 후 자동 제거
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, 3000);
  }
}

class StateManager {
  constructor(elements, apiService) {
    this.currentState = ProcessState.IDLE;
    this.elements = elements; // elements 저장 추가
    this.ui = new UIStateManager(elements);
    this.apiService = apiService;
    this.store = {
      current: {
        uploadId: null,
        analysisId: null,
        tocId: null,
      },
      uploads: new Map(),
      analyses: new Map(),
      tocs: new Map(),
      contents: new Map(),
    };
    this.logger = Logger;
  }

  getAPIService() {
    return this.apiService;
  }

  updateTocAllocation(allocations) {
    const currentToc = this.getCurrentToc();
    if (!currentToc) return;

    // 현재 TOC 데이터에 페이지 할당 정보 업데이트
    currentToc.data.page_allocations = allocations.reduce((acc, allocation) => {
      acc[allocation.id] = allocation.pages;
      return acc;
    }, {});

    // TOC 맵 업데이트
    this.store.tocs.set(currentToc.toc_id, currentToc);
  }
  // 수정 후
  setState(newState, response = null) {
    this.logger.debug("State transition", {
      from: this.currentState,
      to: newState,
      response: response,
    });

    try {
      // 상태 매핑
      if (response) {
        const mappedState = this._mapResponseToState(response);
        newState = mappedState || newState;
      }

      // 상태 검증
      if (!this.validateState(newState)) {
        this.logger.error("Invalid state transition", {
          fromState: this.currentState,
          toState: newState,
        });
        return;
      }

      // 상태 업데이트 및 저장소 업데이트
      this.currentState = newState;
      if (response) {
        this._updateStore(newState, response);
      }

      // UI 업데이트
      this.updateUI();

      // FILE_SELECTED 상태일 때 분석 버튼 활성화
      if (newState === ProcessState.FILE_SELECTED) {
        const analyzeButton = document.getElementById("btn-analyze");
        if (analyzeButton) {
          analyzeButton.classList.remove("hidden");
          analyzeButton.disabled = false;
        }
      }
    } catch (error) {
      this.logger.error("State transition failed", {
        error: error.message,
        state: newState,
      });
      this.currentState = ProcessState.ERROR;
      this.updateUI();
    }
  }

  _determineEndpoint(response) {
    if (!response) {
      Logger.warn("Response is undefined");
      return "default"; // 기본값 반환으로 변경
    }

    try {
      // data 객체 내부도 확인
      const data = response.data || {};

      // 우선순위에 따른 판별
      if (data.file_id || response.file_id) return "upload";
      if (data.analysis_id || response.analysis_id) return "analyze";
      if (data.toc_id || response.toc_id) return "toc";
      if (data.section_id || response.section_id) return "generate";

      // fallback 처리
      Logger.warn("Endpoint type not determined, using default", {
        responseKeys: Object.keys(response),
        dataKeys: Object.keys(data),
      });
      return "default";
    } catch (error) {
      Logger.error("Error determining endpoint", error);
      return "default"; // 에러 발생시 기본값
    }
  }

  _mapResponseToState(response) {
    const endpoint = this._determineEndpoint(response);
    const status = (response?.status || "error").toLowerCase();

    // 성공 상태이고 분석 관련 응답인 경우
    if (status === "success" && response?.message === "문서 분석 완료") {
      return ProcessState.ANALYSIS_COMPLETE;
    }

    // 기존 매핑 로직
    if (endpoint === "upload" && status === "success") {
      return ProcessState.FILE_SELECTED;
    }

    return StatusMapping[status]?.[endpoint] || ProcessState.ERROR;
  }

  validateState(state) {
    // 현재 상태가 undefined인 경우 처리
    const currentState = this.currentState || ProcessState.IDLE;

    // 동일 상태로의 전환은 허용
    if (currentState === state) {
      return true;
    }

    const validTransitions = {
      [ProcessState.IDLE]: [ProcessState.FILE_SELECTED, ProcessState.ERROR],
      [ProcessState.FILE_SELECTED]: [
        ProcessState.ANALYZING,
        ProcessState.ERROR,
        ProcessState.IDLE,
      ],
      [ProcessState.ANALYZING]: [
        ProcessState.ANALYSIS_COMPLETE,
        ProcessState.ERROR,
        ProcessState.FILE_SELECTED,
      ],
      [ProcessState.ANALYSIS_COMPLETE]: [
        ProcessState.GENERATING_TOC,
        ProcessState.ERROR,
        ProcessState.ANALYZING,
      ],
      [ProcessState.GENERATING_TOC]: [
        ProcessState.TOC_COMPLETE,
        ProcessState.ERROR,
        ProcessState.ANALYSIS_COMPLETE,
      ],
      [ProcessState.TOC_COMPLETE]: [
        ProcessState.GENERATING_CONTENT,
        ProcessState.ERROR,
        ProcessState.GENERATING_TOC,
      ],
      [ProcessState.GENERATING_CONTENT]: [
        ProcessState.COMPLETED,
        ProcessState.ERROR,
        ProcessState.TOC_COMPLETE,
      ],
      [ProcessState.COMPLETED]: [ProcessState.IDLE, ProcessState.ERROR],
      [ProcessState.ERROR]: [ProcessState.IDLE, ProcessState.FILE_SELECTED], // ERROR 상태에서 복구 가능하도록
    };

    const isValid = validTransitions[currentState]?.includes(state) ?? false;

    if (!isValid) {
      Logger.warn(
        `Invalid state transition attempted from ${currentState} to ${state}`,
        {
          currentState,
          attemptedState: state,
          validTransitions: validTransitions[currentState],
        }
      );
      return false;
    }

    return true;
  }

  _updateStore(state, response) {
    switch (state) {
      case ProcessState.FILE_SELECTED:
        this.store.uploads.set(response.data.file_id, response);
        this.store.current.uploadId = response.data.file_id;
        break;
      case ProcessState.ANALYSIS_COMPLETE:
        this.store.analyses.set(response.analysis_id, response);
        this.store.current.analysisId = response.analysis_id;
        break;
      case ProcessState.TOC_COMPLETE:
        this.store.tocs.set(response.toc_id, response);
        this.store.current.tocId = response.toc_id;
        break;
      case ProcessState.COMPLETED:
        this.store.contents.set(response.section_id, response);
        break;
      case ProcessState.ERROR:
        Logger.error("Error state", response);
        break;
      case ProcessState.IDLE:
        this.store.current = {
          uploadId: null,
          analysisId: null,
          tocId: null,
        };
        break;
    }
  }

  updateUI() {
    Logger.debug("Updating UI", {
      currentState: this.currentState,
      visibleButtons: this.elements
        ? Object.entries(this.elements)
            .filter(
              ([key, el]) =>
                key.startsWith("btn-") && el && !el.classList.contains("hidden")
            )
            .map(([key]) => key)
        : [],
    });

    // UI 매니저를 통해 업데이트
    this.ui.updateStatusBadge(this.currentState);
    this.ui.updateButtons(this.currentState);
    this.ui.updateSections(this.currentState);
  }

  // Getter 메서드들
  getFileId() {
    return this.store.current.uploadId;
  }

  getAnalysisId() {
    return this.store.current.analysisId;
  }

  getTocId() {
    return this.store.current.tocId;
  }

  getCurrentUpload() {
    return this.store.uploads.get(this.store.current.uploadId);
  }

  getCurrentAnalysis() {
    return this.store.analyses.get(this.store.current.analysisId);
  }

  getCurrentToc() {
    return this.store.tocs.get(this.store.current.tocId);
  }

  getContent(sectionId) {
    return this.store.contents.get(sectionId);
  }
}

// DOM 요소 참조
const elements = {
  uploadForm: document.getElementById("upload-form"),
  fileInput: document.getElementById("file-input"),
  analyzeBtn: document.getElementById("btn-analyze"),
  generateTocBtn: document.getElementById("generate-toc-brn"),
  generateContentBtn: document.getElementById("btn-generate-content"),
  downloadBtn: document.getElementById("btn-download"),
  progressBar: document.getElementById("progress-bar"),
  progressFill: document.getElementById("progress-fill"),
  statusBadge: document.getElementById("status-badge"),
  generatedSections: document.getElementById("generated-sections"),
  notifications: document.getElementById("notifications"),
  fileName: document.getElementById("file-name"),
  fileInfo: document.getElementById("file-info"),
  dropZone: document.getElementById("drop-zone"),
};

// 이벤트 핸들러 함수들
async function handleAnalyzeClick() {
  try {
    const fileId = stateManager.getFileId();
    if (!fileId) {
      showNotification("분석할 파일이 없습니다.", "error");
      return;
    }
    await eventHandlers.handleAnalysisProgress(fileId);
  } catch (error) {
    console.error("분석 중 오류:", error);
    showNotification("분석 실패: " + error.message, "error");
  }
}

async function handleGenerateContentClick() {
  try {
    const tocId = stateManager.getTocId();
    const sectionId = stateManager.getCurrentSection()?.id;
    if (!tocId || !sectionId) {
      showNotification("TOC 또는 섹션 정보가 없습니다.", "error");
      return;
    }
    await eventHandlers.handleContentGeneration(sectionId, tocId);
  } catch (error) {
    console.error("컨텐츠 생성 중 오류:", error);
    showNotification("컨텐츠 생성 실패: " + error.message, "error");
  }
}

async function generateToc() {
  try {
    const analysisId = stateManager.getAnalysisId();
    if (!analysisId) {
      showNotification("분석 결과가 없습니다.", "error");
      return;
    }
    const params = {
      total_pages: 50, // 기본값
      presentation_time: 30, // 기본값
      min_pages_per_section: 2,
    };
    await eventHandlers.handleTocGeneration(analysisId, params);
  } catch (error) {
    console.error("TOC 생성 중 오류:", error);
    showNotification("TOC 생성 실패: " + error.message, "error");
  }
}

// 이벤트 리스너 설정 함수 수정
function setupEventListeners() {
  // 기존 버튼 이벤트 리스너
  if (elements.generateTocBtn) {
    elements.generateTocBtn.addEventListener("click", generateToc);
  }

  if (elements.analyzeBtn) {
    elements.analyzeBtn.addEventListener("click", handleAnalyzeClick);
  }

  if (elements.generateContentBtn) {
    elements.generateContentBtn.addEventListener(
      "click",
      handleGenerateContentClick
    );
  }

  // 드래그 앤 드롭 이벤트 리스너 추가
  const dropZone = document.getElementById("drop-zone");
  if (dropZone) {
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, highlight, false);
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, unhighlight, false);
    });

    dropZone.addEventListener("drop", handleDrop, false);
  }

  // 파일 입력 이벤트 리스너
  const fileInput = document.getElementById("file-input");
  if (fileInput) {
    fileInput.addEventListener("change", handleFileSelect, false);
  }
}

// 드래그 앤 드롭 관련 유틸리티 함수들
function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function highlight(e) {
  const dropZone = document.getElementById("drop-zone");
  if (dropZone) {
    dropZone.classList.add("border-blue-500", "bg-blue-50");
  }
}

function unhighlight(e) {
  const dropZone = document.getElementById("drop-zone");
  if (dropZone) {
    dropZone.classList.remove("border-blue-500", "bg-blue-50");
  }
}

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  if (files.length > 0) {
    eventHandlers.handleFileUpload(files[0]);
  }
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files.length > 0) {
    eventHandlers.handleFileUpload(files[0]);
  }
}

// 초기화
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();

  // StateManager 및 EventHandlers 인스턴스 생성
  const apiService = new APIService();
  stateManager = new StateManager(elements, apiService);
  const modalController = new ModalController();
  eventHandlers = new EventHandlers(stateManager, modalController, elements);
});

// 전역 변수 선언
let stateManager;
let eventHandlers;
