--
-- PostgreSQL database dump
--

\restrict J1YSmLmprZoIH6uHUP9B7ChusYGZX4xKbl8eJLXgbab7wH0k0kXTNAqeudnDLLi

-- Dumped from database version 15.19 (Debian 15.19-1.pgdg13+2)
-- Dumped by pg_dump version 15.19 (Debian 15.19-1.pgdg13+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: client_ready_leads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.client_ready_leads (
    id bigint NOT NULL,
    source_lead_id bigint NOT NULL,
    username text,
    text text,
    msg_timestamp timestamp with time zone,
    lead_type text,
    vertical jsonb,
    geo jsonb,
    payment_methods_mentioned jsonb,
    approved_by text,
    created_at timestamp with time zone DEFAULT now(),
    company text,
    "position" text,
    notes text,
    psp_notified_at timestamp with time zone,
    platform jsonb,
    casino_notified_at timestamp with time zone,
    current_provider text,
    iban_notified_at timestamp with time zone,
    psp_approval_sent_at timestamp with time zone,
    psp_approved boolean,
    psp_approved_by text,
    psp_decided_at timestamp with time zone,
    traffic_source jsonb,
    pricing_model jsonb,
    card_notified_at timestamp with time zone
);


--
-- Name: geo_expansion_map; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.geo_expansion_map (
    parent text NOT NULL,
    child text NOT NULL,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: geo_normalization_map; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.geo_normalization_map (
    raw_value text NOT NULL,
    normalized text NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


--
-- Name: classified_messages_dirty; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.classified_messages_dirty (
    id bigint NOT NULL,
    group_id bigint,
    message_id bigint,
    "timestamp" timestamp with time zone,
    username text,
    text text,
    is_lead boolean,
    confidence double precision,
    lead_type text,
    intent text,
    interest_level text,
    vertical jsonb,
    geo jsonb,
    payment_methods_mentioned jsonb,
    evidence_quote text,
    rationale text,
    glossary_terms_seen jsonb,
    elapsed_ms double precision,
    verdict text,
    judge_reason text,
    db_write_status text DEFAULT 'ok'::text NOT NULL,
    db_write_error text,
    classified_at timestamp with time zone DEFAULT now() NOT NULL,
    manual_approval text,
    asset jsonb,
    network jsonb,
    pay_asset jsonb,
    pay_method jsonb,
    amount double precision,
    min double precision,
    max double precision,
    company text,
    pipeline text,
    "position" text,
    traffic_source jsonb,
    pricing_model jsonb,
    traffic_notified_at timestamp with time zone,
    traffic_sent_at timestamp with time zone
);


--
-- Name: COLUMN classified_messages_dirty.traffic_notified_at; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.classified_messages_dirty.traffic_notified_at IS 'When the row was delivered to its traffic feed. NULL = still pending, which is exactly what the boot catch-up replays.';


--
-- Name: COLUMN classified_messages_dirty.traffic_sent_at; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.classified_messages_dirty.traffic_sent_at IS 'When the row was actually delivered to its traffic feed (migration 031). NULL with traffic_notified_at set = handled without a send (duplicate suppressed or skipped by the backfill policy).';


--
-- Name: payment_options; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.payment_options (
    token text NOT NULL,
    display_name text NOT NULL,
    category text,
    aliases text[] DEFAULT '{}'::text[] NOT NULL,
    sort_order integer DEFAULT 0,
    parent_token text,
    is_local_method boolean DEFAULT false NOT NULL,
    local_method_country text[] DEFAULT '{}'::text[] NOT NULL,
    CONSTRAINT payment_options_parent_not_self CHECK (((parent_token IS NULL) OR (parent_token <> token)))
);


--
-- Name: banned_usernames; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.banned_usernames (
    username text NOT NULL,
    added_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: classified_messages_dirty_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.classified_messages_dirty_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: classified_messages_dirty_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.classified_messages_dirty_id_seq OWNED BY public.classified_messages_dirty.id;


--
-- Name: client_ready_leads_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.client_ready_leads_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: client_ready_leads_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.client_ready_leads_id_seq OWNED BY public.client_ready_leads.id;


--
-- Name: dedup_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dedup_log (
    username text NOT NULL,
    text_hash text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: geo_unknown_values; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.geo_unknown_values (
    raw_value text NOT NULL,
    message_id integer NOT NULL,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: payment_geo_map; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.payment_geo_map (
    payment_method text NOT NULL,
    geo_code text NOT NULL,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: payment_unknown_values; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.payment_unknown_values (
    raw_value text,
    message_id bigint,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: classified_messages_dirty id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classified_messages_dirty ALTER COLUMN id SET DEFAULT nextval('public.classified_messages_dirty_id_seq'::regclass);


--
-- Name: client_ready_leads id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.client_ready_leads ALTER COLUMN id SET DEFAULT nextval('public.client_ready_leads_id_seq'::regclass);


--
-- Name: banned_usernames banned_usernames_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.banned_usernames
    ADD CONSTRAINT banned_usernames_pkey PRIMARY KEY (username);


--
-- Name: classified_messages_dirty classified_messages_dirty_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classified_messages_dirty
    ADD CONSTRAINT classified_messages_dirty_pkey PRIMARY KEY (id);


--
-- Name: client_ready_leads client_ready_leads_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.client_ready_leads
    ADD CONSTRAINT client_ready_leads_pkey PRIMARY KEY (id);


--
-- Name: client_ready_leads client_ready_leads_source_lead_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.client_ready_leads
    ADD CONSTRAINT client_ready_leads_source_lead_id_key UNIQUE (source_lead_id);


--
-- Name: dedup_log dedup_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dedup_log
    ADD CONSTRAINT dedup_log_pkey PRIMARY KEY (username);


--
-- Name: geo_expansion_map geo_expansion_map_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.geo_expansion_map
    ADD CONSTRAINT geo_expansion_map_pkey PRIMARY KEY (parent, child);


--
-- Name: geo_normalization_map geo_normalization_map_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.geo_normalization_map
    ADD CONSTRAINT geo_normalization_map_pkey PRIMARY KEY (raw_value);


--
-- Name: geo_unknown_values geo_unknown_values_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.geo_unknown_values
    ADD CONSTRAINT geo_unknown_values_pkey PRIMARY KEY (raw_value, message_id);


--
-- Name: payment_geo_map payment_geo_map_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payment_geo_map
    ADD CONSTRAINT payment_geo_map_pkey PRIMARY KEY (payment_method, geo_code);


--
-- Name: payment_options payment_options_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payment_options
    ADD CONSTRAINT payment_options_pkey PRIMARY KEY (token);


--
-- Name: classified_messages_dirty uq_group_message_leadtype; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classified_messages_dirty
    ADD CONSTRAINT uq_group_message_leadtype UNIQUE (group_id, message_id, lead_type);


--
-- Name: payment_unknown_values uq_payment_unknown; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payment_unknown_values
    ADD CONSTRAINT uq_payment_unknown UNIQUE (raw_value, message_id);


--
-- Name: ix_cmd_traffic_approved_unsent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_cmd_traffic_approved_unsent ON public.classified_messages_dirty USING btree (verdict, id) WHERE ((manual_approval = 'True'::text) AND (traffic_sent_at IS NULL));


--
-- Name: ix_cmd_traffic_in_review; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_cmd_traffic_in_review ON public.classified_messages_dirty USING btree (verdict) WHERE (manual_approval = 'pending'::text);


--
-- Name: ix_cmd_traffic_pending; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_cmd_traffic_pending ON public.classified_messages_dirty USING btree (verdict, id) WHERE (traffic_notified_at IS NULL);


--
-- Name: ix_cmd_traffic_sent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_cmd_traffic_sent ON public.classified_messages_dirty USING btree (verdict, traffic_sent_at) WHERE (traffic_sent_at IS NOT NULL);


--
-- Name: ix_payment_options_local; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_payment_options_local ON public.payment_options USING btree (is_local_method);


--
-- Name: ix_payment_options_parent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_payment_options_parent ON public.payment_options USING btree (parent_token);


--
-- Name: banned_usernames banned_usernames_notify; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER banned_usernames_notify AFTER INSERT OR DELETE ON public.banned_usernames FOR EACH ROW EXECUTE FUNCTION public.notify_banned_usernames_changed();


--
-- Name: client_ready_leads card_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER card_lead_on_insert AFTER INSERT ON public.client_ready_leads FOR EACH ROW WHEN ((new.lead_type = ANY (ARRAY['card_seeker'::text, 'card_provider'::text]))) EXECUTE FUNCTION public.notify_card_lead();


--
-- Name: client_ready_leads card_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER card_lead_on_update AFTER UPDATE ON public.client_ready_leads FOR EACH ROW WHEN (((new.lead_type = ANY (ARRAY['card_seeker'::text, 'card_provider'::text])) AND (old.lead_type IS DISTINCT FROM new.lead_type))) EXECUTE FUNCTION public.notify_card_lead();


--
-- Name: client_ready_leads casino_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER casino_lead_on_insert AFTER INSERT ON public.client_ready_leads FOR EACH ROW WHEN ((new.lead_type = 'seeking_casino_platform'::text)) EXECUTE FUNCTION public.notify_casino_lead();


--
-- Name: client_ready_leads casino_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER casino_lead_on_update AFTER UPDATE ON public.client_ready_leads FOR EACH ROW WHEN (((new.lead_type = 'seeking_casino_platform'::text) AND (old.lead_type IS DISTINCT FROM new.lead_type))) EXECUTE FUNCTION public.notify_casino_lead();


--
-- Name: classified_messages_dirty crypto_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER crypto_lead_on_insert AFTER INSERT ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'REAL_LEAD'::text) AND (new.lead_type = 'crypto_buyer'::text))) EXECUTE FUNCTION public.notify_crypto_lead();


--
-- Name: classified_messages_dirty crypto_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER crypto_lead_on_update AFTER UPDATE ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'REAL_LEAD'::text) AND (new.lead_type = 'crypto_buyer'::text) AND (old.verdict IS DISTINCT FROM new.verdict))) EXECUTE FUNCTION public.notify_crypto_lead();


--
-- Name: client_ready_leads iban_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER iban_lead_on_insert AFTER INSERT ON public.client_ready_leads FOR EACH ROW WHEN ((new.lead_type = 'seeking_iban'::text)) EXECUTE FUNCTION public.notify_iban_lead();


--
-- Name: client_ready_leads iban_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER iban_lead_on_update AFTER UPDATE ON public.client_ready_leads FOR EACH ROW WHEN (((new.lead_type = 'seeking_iban'::text) AND (old.lead_type IS DISTINCT FROM new.lead_type))) EXECUTE FUNCTION public.notify_iban_lead();


--
-- Name: client_ready_leads psp_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER psp_lead_on_insert AFTER INSERT ON public.client_ready_leads FOR EACH ROW WHEN ((new.lead_type = 'psp_provider'::text)) EXECUTE FUNCTION public.notify_psp_lead();


--
-- Name: client_ready_leads psp_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER psp_lead_on_update AFTER UPDATE ON public.client_ready_leads FOR EACH ROW WHEN (((new.lead_type = 'psp_provider'::text) AND (old.lead_type IS DISTINCT FROM new.lead_type))) EXECUTE FUNCTION public.notify_psp_lead();


--
-- Name: classified_messages_dirty psp_provider_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER psp_provider_lead_on_insert AFTER INSERT ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'REAL_PROVIDER'::text) AND (new.lead_type = 'psp_provider'::text))) EXECUTE FUNCTION public.notify_psp_provider_lead();


--
-- Name: classified_messages_dirty psp_provider_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER psp_provider_lead_on_update AFTER UPDATE ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'REAL_PROVIDER'::text) AND (new.lead_type = 'psp_provider'::text) AND (old.verdict IS DISTINCT FROM new.verdict))) EXECUTE FUNCTION public.notify_psp_provider_lead();


--
-- Name: classified_messages_dirty real_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER real_lead_on_insert AFTER INSERT ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'REAL_LEAD'::text) AND (new.lead_type IS DISTINCT FROM 'psp_provider'::text) AND (new.lead_type IS DISTINCT FROM 'seeking_casino_platform'::text) AND (new.lead_type IS DISTINCT FROM 'seeking_iban'::text) AND (new.lead_type IS DISTINCT FROM 'crypto_buyer'::text) AND (new.lead_type IS DISTINCT FROM 'card_seeker'::text) AND (new.lead_type IS DISTINCT FROM 'card_provider'::text))) EXECUTE FUNCTION public.notify_real_lead();


--
-- Name: classified_messages_dirty real_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER real_lead_on_update AFTER UPDATE ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'REAL_LEAD'::text) AND (new.lead_type IS DISTINCT FROM 'psp_provider'::text) AND (new.lead_type IS DISTINCT FROM 'seeking_casino_platform'::text) AND (new.lead_type IS DISTINCT FROM 'seeking_iban'::text) AND (new.lead_type IS DISTINCT FROM 'crypto_buyer'::text) AND (new.lead_type IS DISTINCT FROM 'card_seeker'::text) AND (new.lead_type IS DISTINCT FROM 'card_provider'::text) AND (old.verdict IS DISTINCT FROM new.verdict))) EXECUTE FUNCTION public.notify_real_lead();


--
-- Name: classified_messages_dirty traffic_lead_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER traffic_lead_on_insert AFTER INSERT ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'TRAFFIC_LEAD'::text) AND (new.lead_type = 'seeking_traffic'::text))) EXECUTE FUNCTION public.notify_traffic_lead();


--
-- Name: classified_messages_dirty traffic_lead_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER traffic_lead_on_update AFTER UPDATE ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'TRAFFIC_LEAD'::text) AND (new.lead_type = 'seeking_traffic'::text) AND (old.verdict IS DISTINCT FROM new.verdict))) EXECUTE FUNCTION public.notify_traffic_lead();


--
-- Name: classified_messages_dirty traffic_provider_on_insert; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER traffic_provider_on_insert AFTER INSERT ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'TRAFFIC_PROVIDER'::text) AND (new.lead_type = 'traffic_provider'::text))) EXECUTE FUNCTION public.notify_traffic_provider_lead();


--
-- Name: classified_messages_dirty traffic_provider_on_update; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER traffic_provider_on_update AFTER UPDATE ON public.classified_messages_dirty FOR EACH ROW WHEN (((new.verdict = 'TRAFFIC_PROVIDER'::text) AND (new.lead_type = 'traffic_provider'::text) AND (old.verdict IS DISTINCT FROM new.verdict))) EXECUTE FUNCTION public.notify_traffic_provider_lead();


--
-- Name: banned_usernames trg_banned_username_added; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_banned_username_added AFTER INSERT ON public.banned_usernames FOR EACH ROW EXECUTE FUNCTION public.notify_banned_username_added();


--
-- Name: payment_options payment_options_parent_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.payment_options
    ADD CONSTRAINT payment_options_parent_fk FOREIGN KEY (parent_token) REFERENCES public.payment_options(token) ON DELETE SET NULL;


--
-- PostgreSQL database dump complete
--

\unrestrict J1YSmLmprZoIH6uHUP9B7ChusYGZX4xKbl8eJLXgbab7wH0k0kXTNAqeudnDLLi

