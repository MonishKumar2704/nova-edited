/**
 * Nova AI - dynamic action / result-card rendering (Phase 4).
 *
 * Renders whatever `{"type", "data", "actions": [{"id","label",
 * "requires_confirmation"}]}` cards the backend returns (master spec
 * section 10). Card *type* determines the fields shown (currently just
 * `youtube_video`; a `gmail_message` type in Phase 6/7 plugs in here with
 * zero changes to the agent/YouTube code). Action *dispatch* is left to
 * the caller via `onAction(actionId, card)` - this module never itself
 * decides what "play" or "queue" mean, so the same renderer works for
 * cards coming from a voice command, a text search, or (later) Gmail.
 */
(function (global) {
  "use strict";

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML;
  }

  function renderYoutubeVideoCard(card) {
    const v = card.data || {};
    const thumb = v.thumbnail_url
      ? `<img class="nova-card-thumb" src="${escapeHtml(v.thumbnail_url)}" alt="" loading="lazy">`
      : `<div class="nova-card-thumb nova-card-thumb--empty"></div>`;

    return `
      <div class="nova-card nova-card--youtube_video" data-video-id="${escapeHtml(v.video_id)}">
        ${thumb}
        <div class="nova-card-body">
          <div class="nova-card-title">${escapeHtml(v.title)}</div>
          <div class="nova-card-subtitle">${escapeHtml(v.channel_title)}</div>
        </div>
        <div class="nova-card-actions"></div>
      </div>
    `;
  }

  // Phase 6/7: minimal Gmail card renderers. Full email reader/composer UI
  // (Phase 14) replaces these with richer components; these exist now so
  // Phase 6/7 message/thread/draft cards are visible rather than falling
  // through to "Unsupported card type".
  function renderGmailMessageCard(card) {
    const m = card.data || {};
    const unreadClass = m.is_unread ? " nova-card--unread" : "";
    return `
      <div class="nova-card nova-card--gmail_message${unreadClass}" data-message-id="${escapeHtml(m.message_id)}">
        <div class="nova-card-body">
          <div class="nova-card-title">${escapeHtml(m.subject || "(no subject)")}</div>
          <div class="nova-card-subtitle">${escapeHtml(m.from)}</div>
          <div class="nova-card-snippet">${escapeHtml(m.snippet)}</div>
        </div>
        <div class="nova-card-actions"></div>
      </div>
    `;
  }

  function renderGmailThreadCard(card) {
    const t = card.data || {};
    return `
      <div class="nova-card nova-card--gmail_thread" data-thread-id="${escapeHtml(t.thread_id)}">
        <div class="nova-card-body">
          <div class="nova-card-snippet">${escapeHtml(t.snippet)}</div>
          <div class="nova-card-subtitle">${escapeHtml(t.message_count)} message(s)</div>
        </div>
        <div class="nova-card-actions"></div>
      </div>
    `;
  }

  function renderGmailDraftCard(card) {
    const d = card.data || {};
    const m = d.message || {};
    return `
      <div class="nova-card nova-card--gmail_draft" data-draft-id="${escapeHtml(d.draft_id)}">
        <div class="nova-card-body">
          <div class="nova-card-title">${escapeHtml(m.subject || "(no subject)")}</div>
          <div class="nova-card-subtitle">${escapeHtml(m.to || "")}</div>
        </div>
        <div class="nova-card-actions"></div>
      </div>
    `;
  }

  const CARD_RENDERERS = {
    youtube_video: renderYoutubeVideoCard,
    gmail_message: renderGmailMessageCard,
    gmail_thread: renderGmailThreadCard,
    gmail_draft: renderGmailDraftCard,
  };

  class DynamicUI {
    /**
     * @param {Object} options
     * @param {HTMLElement} options.container - where cards are rendered.
     * @param {Function} options.onAction - (actionId, card) => void, called
     *   after any `requires_confirmation` action has been confirmed.
     * @param {Function} [options.confirm] - (card, action) => Promise<boolean>,
     *   defaults to `window.confirm`. Override for a nicer confirmation UI.
     */
    constructor({ container, onAction, confirm }) {
      this._container = container;
      this._onAction = onAction || function () {};
      this._confirm =
        confirm ||
        function (card, action) {
          return Promise.resolve(global.confirm(`${action.label}?`));
        };
    }

    /** Render a list of cards, replacing whatever was previously shown. */
    renderCards(cards) {
      this._container.innerHTML = "";
      (cards || []).forEach((card) => this._container.appendChild(this._buildCardElement(card)));
    }

    /** Render (or re-render) a single card, e.g. "now playing". */
    renderSingleCard(card) {
      this.renderCards(card ? [card] : []);
    }

    _buildCardElement(card) {
      const renderer = CARD_RENDERERS[card.type];
      const wrapper = document.createElement("div");
      wrapper.className = "nova-card-wrapper";

      if (!renderer) {
        // Unknown/future card type: fail soft, not silently (master spec
        // section 13) - show something rather than nothing.
        wrapper.innerHTML = `<div class="nova-card nova-card--unknown">Unsupported card type: ${escapeHtml(
          card.type
        )}</div>`;
        return wrapper;
      }

      wrapper.innerHTML = renderer(card);
      const actionsEl = wrapper.querySelector(".nova-card-actions");
      if (actionsEl) {
        (card.actions || []).forEach((action) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "nova-action-btn";
          btn.textContent = action.label;
          btn.dataset.actionId = action.id;
          btn.addEventListener("click", () => this._dispatch(action, card));
          actionsEl.appendChild(btn);
        });
      }
      return wrapper;
    }

    async _dispatch(action, card) {
      if (action.requires_confirmation) {
        const confirmed = await this._confirm(card, action);
        if (!confirmed) return;
      }
      this._onAction(action.id, card);
    }
  }

  global.NovaDynamicUI = DynamicUI;
})(window);
