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

  // Gmail generates a message/thread's `snippet` field itself (a
  // plain-text preview derived from the body) but leaves HTML entities
  // in it un-decoded - a well-documented Gmail API quirk, not something
  // Nova's own extraction introduces (see `app.integrations.gmail_api`,
  // which reads `snippet` straight off the API response). So an
  // apostrophe in the original message shows up in `snippet` as the
  // literal six characters `&#39;`, not a real apostrophe. Passed
  // straight through `escapeHtml()`, that literal `&` gets re-escaped to
  // `&amp;`, and the browser renders the still-encoded `&#39;` as visible
  // text instead of an apostrophe. Decoding first (via a detached
  // `<textarea>`, which parses entities without executing any markup)
  // and re-escaping after fixes that without reopening any injection risk.
  function decodeHtmlEntities(str) {
    if (!str) return str;
    const textarea = document.createElement("textarea");
    textarea.innerHTML = String(str);
    return textarea.value;
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
          <div class="nova-card-snippet">${escapeHtml(decodeHtmlEntities(m.snippet))}</div>
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
          <div class="nova-card-snippet">${escapeHtml(decodeHtmlEntities(t.snippet))}</div>
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

  // Which `data-*` attribute (already stamped on every rendered card
  // above) identifies a given card type, and which key of `card.data`
  // holds that id. Used by `updateCard()` below to find an
  // already-rendered card without re-rendering the whole list.
  const CARD_ID_ATTR = {
    youtube_video: { attr: "video-id", dataKey: "video_id" },
    gmail_message: { attr: "message-id", dataKey: "message_id" },
    gmail_thread: { attr: "thread-id", dataKey: "thread_id" },
    gmail_draft: { attr: "draft-id", dataKey: "draft_id" },
  };

  function cssEscapeAttrValue(value) {
    // `CSS.escape` isn't polyfilled everywhere this file might run, but is
    // standard in every browser Nova actually targets; fall back to a
    // conservative manual escape if it's ever missing.
    if (global.CSS && typeof global.CSS.escape === "function") return global.CSS.escape(value);
    return String(value).replace(/["\\]/g, "\\$&");
  }

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

    /**
     * Re-render one already-displayed card in place, e.g. after a Gmail
     * message action (mark read, star, archive...) returns the message's
     * updated state (master spec section 10 / Task 08: "avoid unnecessary
     * full-page reloads"). Finds the currently rendered node via the
     * type-specific `data-*` id attribute every card already carries and
     * swaps only that one element - the rest of whatever list is showing
     * (an inbox, a search result, a chat reply) is untouched.
     *
     * Returns `true` if a matching card was found and updated, `false` if
     * this card isn't currently displayed (e.g. it came from a different
     * list) - callers should fall back to a toast-only confirmation in
     * that case rather than assuming the UI changed.
     */
    updateCard(card) {
      const idInfo = card && CARD_ID_ATTR[card.type];
      const idValue = idInfo && card.data ? card.data[idInfo.dataKey] : null;
      if (!idInfo || idValue == null) return false;

      const existing = this._container.querySelector(`[data-${idInfo.attr}="${cssEscapeAttrValue(String(idValue))}"]`);
      const wrapper = existing && existing.closest(".nova-card-wrapper");
      if (!wrapper) return false;

      wrapper.replaceWith(this._buildCardElement(card));
      return true;
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
