/**
 * Nova AI - reusable YouTube IFrame Player API wrapper (Phase 4).
 *
 * Wraps the *official* YouTube IFrame Player API (master spec section 18:
 * "use official YouTube APIs/player capabilities, never HTML scraping").
 * Exposes a small, framework-agnostic surface (play/pause/stop/seek/
 * volume/mute/playback-rate/queue) plus an event-callback interface so the
 * page (or a future Phase 14 frontend rewrite) never has to touch the raw
 * `YT.Player` object directly.
 *
 * This file intentionally has ZERO knowledge of dynamic action cards,
 * agent responses, or the DOM outside the player container it's given -
 * that glue lives in `dynamic-actions.js` / `index.html` (master spec
 * section 10: "the frontend must not contain duplicated business logic").
 */
(function (global) {
  "use strict";

  const STATE_NAMES = {
    "-1": "UNSTARTED",
    0: "ENDED",
    1: "PLAYING",
    2: "PAUSED",
    3: "BUFFERING",
    5: "CUED",
  };

  const ERROR_NAMES = {
    2: "INVALID_PARAMETER",
    5: "HTML5_PLAYER_ERROR",
    100: "VIDEO_NOT_FOUND",
    101: "EMBED_NOT_ALLOWED",
    150: "EMBED_NOT_ALLOWED",
  };

  // The IFrame API script calls this exact global function name once it has
  // loaded. Multiple NovaYouTubePlayer instances share one script load and
  // queue their init work until the API is actually ready.
  let apiReadyPromise = null;

  function loadIframeApi() {
    if (apiReadyPromise) return apiReadyPromise;

    apiReadyPromise = new Promise((resolve) => {
      if (global.YT && global.YT.Player) {
        resolve(global.YT);
        return;
      }
      const previous = global.onYouTubeIframeAPIReady;
      global.onYouTubeIframeAPIReady = function () {
        if (typeof previous === "function") previous();
        resolve(global.YT);
      };
      if (!document.getElementById("nova-yt-iframe-api")) {
        const tag = document.createElement("script");
        tag.id = "nova-yt-iframe-api";
        tag.src = "https://www.youtube.com/iframe_api";
        document.head.appendChild(tag);
      }
    });
    return apiReadyPromise;
  }

  class NovaYouTubePlayer {
    /**
     * @param {Object} options
     * @param {string} options.elementId - id of the container element the
     *   IFrame player replaces.
     * @param {Function} [options.onStateChange] - (stateName, stateCode) => void
     * @param {Function} [options.onError] - (errorName, errorCode) => void
     * @param {Function} [options.onReady] - () => void
     * @param {Function} [options.onAutoplayBlocked] - () => void, fired
     *   when we asked for autoplay and the player never reached PLAYING -
     *   so the UI can tell the truth instead of assuming playback started
     *   (master spec Phase 4: "do not falsely report playback").
     */
    constructor(options) {
      this._elementId = options.elementId;
      this._onStateChange = options.onStateChange || function () {};
      this._onError = options.onError || function () {};
      this._onReady = options.onReady || function () {};
      this._onAutoplayBlocked = options.onAutoplayBlocked || function () {};

      this._player = null;
      this._ready = false;
      this._lastKnownState = "UNSTARTED";
      this._queue = []; // queue foundation (Phase 4); playlist tools land Phase 5
      this._autoplayCheckTimer = null;

      this._initPromise = loadIframeApi().then((YT) => this._createPlayer(YT));
    }

    _createPlayer(YT) {
      return new Promise((resolve) => {
        this._player = new YT.Player(this._elementId, {
          height: "100%",
          width: "100%",
          playerVars: {
            playsinline: 1,
            rel: 0,
            modestbranding: 1,
          },
          events: {
            onReady: () => {
              this._ready = true;
              this._onReady();
              resolve(this);
            },
            onStateChange: (event) => this._handleStateChange(event),
            onError: (event) => this._handleError(event),
          },
        });
      });
    }

    _handleStateChange(event) {
      const name = STATE_NAMES[String(event.data)] || "UNKNOWN";
      this._lastKnownState = name;

      if (name === "PLAYING" && this._autoplayCheckTimer) {
        clearTimeout(this._autoplayCheckTimer);
        this._autoplayCheckTimer = null;
      }

      if (name === "ENDED") {
        this._playNextInQueue();
      }

      this._onStateChange(name, event.data);
    }

    _handleError(event) {
      const name = ERROR_NAMES[String(event.data)] || "UNKNOWN_PLAYER_ERROR";
      this._onError(name, event.data);
    }

    /** Resolves once the underlying YT.Player is constructed and ready. */
    ready() {
      return this._initPromise;
    }

    /**
     * Load and play a video by ID. `autoplay` defaults to true but is only
     * ever *reported* as playing once the PLAYING state actually fires -
     * if the browser blocks autoplay, `onAutoplayBlocked` fires instead
     * after a short grace period so the UI can prompt for a manual tap.
     */
    async loadVideo(videoId, { autoplay = true } = {}) {
      await this.ready();
      if (autoplay) {
        this._player.loadVideoById(videoId);
        this._armAutoplayCheck();
      } else {
        this._player.cueVideoById(videoId);
      }
    }

    _armAutoplayCheck() {
      if (this._autoplayCheckTimer) clearTimeout(this._autoplayCheckTimer);
      this._autoplayCheckTimer = setTimeout(() => {
        if (this._lastKnownState !== "PLAYING") {
          this._onAutoplayBlocked();
        }
      }, 1500);
    }

    async play() {
      await this.ready();
      this._player.playVideo();
    }

    async pause() {
      await this.ready();
      this._player.pauseVideo();
    }

    /** Stop playback entirely (distinct from pause - resets playback position). */
    async stop() {
      await this.ready();
      this._player.stopVideo();
    }

    async seekTo(seconds, { allowSeekAhead = true } = {}) {
      await this.ready();
      this._player.seekTo(seconds, allowSeekAhead);
    }

    async getCurrentTime() {
      await this.ready();
      return this._player.getCurrentTime();
    }

    async getDuration() {
      await this.ready();
      return this._player.getDuration();
    }

    async setVolume(volume0to100) {
      await this.ready();
      this._player.setVolume(Math.max(0, Math.min(100, volume0to100)));
    }

    async getVolume() {
      await this.ready();
      return this._player.getVolume();
    }

    async mute() {
      await this.ready();
      this._player.mute();
    }

    async unmute() {
      await this.ready();
      this._player.unMute();
    }

    async isMuted() {
      await this.ready();
      return this._player.isMuted();
    }

    async setPlaybackRate(rate) {
      await this.ready();
      this._player.setPlaybackRate(rate);
    }

    async getPlaybackRate() {
      await this.ready();
      return this._player.getPlaybackRate();
    }

    async getAvailablePlaybackRates() {
      await this.ready();
      return this._player.getAvailablePlaybackRates();
    }

    getState() {
      return this._lastKnownState;
    }

    // --- Queue foundation (Phase 4) -----------------------------------
    // Deliberately simple (array + index) - real playlist persistence via
    // the YouTube Data API (server-side playlists) is Phase 5's job. This
    // is just enough for "add to queue" / "play next" to work today.

    enqueue(videoId, meta) {
      this._queue.push({ videoId, meta: meta || {} });
    }

    clearQueue() {
      this._queue = [];
    }

    getQueue() {
      return this._queue.slice();
    }

    async _playNextInQueue() {
      const next = this._queue.shift();
      if (next) {
        await this.loadVideo(next.videoId, { autoplay: true });
      }
    }

    async playNext() {
      await this._playNextInQueue();
    }
  }

  global.NovaYouTubePlayer = NovaYouTubePlayer;
})(window);
