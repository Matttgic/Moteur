export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      tennis_feature_snapshots: {
        Row: {
          break_pct: number | null
          captured_at: string
          data_quality: number | null
          elo: number | null
          fatigue_score: number | null
          feature_version: string
          form_score: number | null
          hold_pct: number | null
          id: number
          match_id: string
          physical_risk: number | null
          player_id: string
          ranking: number | null
          surface_elo: number | null
        }
        Insert: {
          break_pct?: number | null
          captured_at: string
          data_quality?: number | null
          elo?: number | null
          fatigue_score?: number | null
          feature_version: string
          form_score?: number | null
          hold_pct?: number | null
          id?: number
          match_id: string
          physical_risk?: number | null
          player_id: string
          ranking?: number | null
          surface_elo?: number | null
        }
        Update: {
          break_pct?: number | null
          captured_at?: string
          data_quality?: number | null
          elo?: number | null
          fatigue_score?: number | null
          feature_version?: string
          form_score?: number | null
          hold_pct?: number | null
          id?: number
          match_id?: string
          physical_risk?: number | null
          player_id?: string
          ranking?: number | null
          surface_elo?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "tennis_feature_snapshots_match_id_fkey"
            columns: ["match_id"]
            isOneToOne: false
            referencedRelation: "tennis_matches"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tennis_feature_snapshots_player_id_fkey"
            columns: ["player_id"]
            isOneToOne: false
            referencedRelation: "tennis_players"
            referencedColumns: ["id"]
          },
        ]
      }
      tennis_ingestion_runs: {
        Row: {
          details: Json
          fetched_matches: number
          finished_at: string | null
          id: string
          job_name: string
          processed_matches: number
          provider_requests: number
          skipped_matches: number
          started_at: string
          status: string
          tour: string | null
          window_from: string | null
          window_to: string | null
        }
        Insert: {
          details?: Json
          fetched_matches?: number
          finished_at?: string | null
          id?: string
          job_name: string
          processed_matches?: number
          provider_requests?: number
          skipped_matches?: number
          started_at?: string
          status: string
          tour?: string | null
          window_from?: string | null
          window_to?: string | null
        }
        Update: {
          details?: Json
          fetched_matches?: number
          finished_at?: string | null
          id?: string
          job_name?: string
          processed_matches?: number
          provider_requests?: number
          skipped_matches?: number
          started_at?: string
          status?: string
          tour?: string | null
          window_from?: string | null
          window_to?: string | null
        }
        Relationships: []
      }
      tennis_matches: {
        Row: {
          completed_at: string | null
          created_at: string
          final_score: Json | null
          id: string
          indoor: boolean
          outcome: string | null
          player_a_id: string
          player_b_id: string
          provider_match_id: string | null
          provider_updated_at: string | null
          round: string | null
          scheduled_at: string
          status: string
          surface: string
          tour: string
          tournament: string
          updated_at: string
          winner_id: string | null
        }
        Insert: {
          completed_at?: string | null
          created_at?: string
          final_score?: Json | null
          id?: string
          indoor?: boolean
          outcome?: string | null
          player_a_id: string
          player_b_id: string
          provider_match_id?: string | null
          provider_updated_at?: string | null
          round?: string | null
          scheduled_at: string
          status?: string
          surface: string
          tour: string
          tournament: string
          updated_at?: string
          winner_id?: string | null
        }
        Update: {
          completed_at?: string | null
          created_at?: string
          final_score?: Json | null
          id?: string
          indoor?: boolean
          outcome?: string | null
          player_a_id?: string
          player_b_id?: string
          provider_match_id?: string | null
          provider_updated_at?: string | null
          round?: string | null
          scheduled_at?: string
          status?: string
          surface?: string
          tour?: string
          tournament?: string
          updated_at?: string
          winner_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "tennis_matches_player_a_id_fkey"
            columns: ["player_a_id"]
            isOneToOne: false
            referencedRelation: "tennis_players"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tennis_matches_player_b_id_fkey"
            columns: ["player_b_id"]
            isOneToOne: false
            referencedRelation: "tennis_players"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tennis_matches_winner_id_fkey"
            columns: ["winner_id"]
            isOneToOne: false
            referencedRelation: "tennis_players"
            referencedColumns: ["id"]
          },
        ]
      }
      tennis_model_runs: {
        Row: {
          artifact_uri: string | null
          avg_clv: number | null
          brier_score: number | null
          created_at: string
          id: string
          log_loss: number | null
          model_name: string
          model_version: string
          roi: number | null
          sample_size: number | null
          tour: string
          trained_through: string | null
          validation_method: string
        }
        Insert: {
          artifact_uri?: string | null
          avg_clv?: number | null
          brier_score?: number | null
          created_at?: string
          id?: string
          log_loss?: number | null
          model_name: string
          model_version: string
          roi?: number | null
          sample_size?: number | null
          tour: string
          trained_through?: string | null
          validation_method?: string
        }
        Update: {
          artifact_uri?: string | null
          avg_clv?: number | null
          brier_score?: number | null
          created_at?: string
          id?: string
          log_loss?: number | null
          model_name?: string
          model_version?: string
          roi?: number | null
          sample_size?: number | null
          tour?: string
          trained_through?: string | null
          validation_method?: string
        }
        Relationships: []
      }
      tennis_odds_snapshots: {
        Row: {
          bookmaker: string
          captured_at: string
          id: number
          is_closing: boolean
          market: string
          match_id: string
          player_a_odds: number
          player_b_odds: number
        }
        Insert: {
          bookmaker: string
          captured_at: string
          id?: number
          is_closing?: boolean
          market?: string
          match_id: string
          player_a_odds: number
          player_b_odds: number
        }
        Update: {
          bookmaker?: string
          captured_at?: string
          id?: number
          is_closing?: boolean
          market?: string
          match_id?: string
          player_a_odds?: number
          player_b_odds?: number
        }
        Relationships: [
          {
            foreignKeyName: "tennis_odds_snapshots_match_id_fkey"
            columns: ["match_id"]
            isOneToOne: false
            referencedRelation: "tennis_matches"
            referencedColumns: ["id"]
          },
        ]
      }
      tennis_outcomes: {
        Row: {
          match_id: string
          retired: boolean
          score: string | null
          settled_at: string
          walkover: boolean
          winner_id: string
        }
        Insert: {
          match_id: string
          retired?: boolean
          score?: string | null
          settled_at: string
          walkover?: boolean
          winner_id: string
        }
        Update: {
          match_id?: string
          retired?: boolean
          score?: string | null
          settled_at?: string
          walkover?: boolean
          winner_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "tennis_outcomes_match_id_fkey"
            columns: ["match_id"]
            isOneToOne: true
            referencedRelation: "tennis_matches"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tennis_outcomes_winner_id_fkey"
            columns: ["winner_id"]
            isOneToOne: false
            referencedRelation: "tennis_players"
            referencedColumns: ["id"]
          },
        ]
      }
      tennis_player_states: {
        Row: {
          carpet_elo: number
          clay_elo: number
          data_quality: number
          elo: number
          feature_version: string
          grass_elo: number
          hard_elo: number
          last_match_at: string | null
          player_id: string
          recent_match_dates: Json
          recent_results: Json
          return_breaks: number
          return_games: number
          service_games: number
          service_holds: number
          service_sample_matches: number
          source: string
          state_as_of: string
          tour: string
          updated_at: string
        }
        Insert: {
          carpet_elo?: number
          clay_elo?: number
          data_quality?: number
          elo?: number
          feature_version?: string
          grass_elo?: number
          hard_elo?: number
          last_match_at?: string | null
          player_id: string
          recent_match_dates?: Json
          recent_results?: Json
          return_breaks?: number
          return_games?: number
          service_games?: number
          service_holds?: number
          service_sample_matches?: number
          source?: string
          state_as_of?: string
          tour: string
          updated_at?: string
        }
        Update: {
          carpet_elo?: number
          clay_elo?: number
          data_quality?: number
          elo?: number
          feature_version?: string
          grass_elo?: number
          hard_elo?: number
          last_match_at?: string | null
          player_id?: string
          recent_match_dates?: Json
          recent_results?: Json
          return_breaks?: number
          return_games?: number
          service_games?: number
          service_holds?: number
          service_sample_matches?: number
          source?: string
          state_as_of?: string
          tour?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "tennis_player_states_player_id_fkey"
            columns: ["player_id"]
            isOneToOne: true
            referencedRelation: "tennis_players"
            referencedColumns: ["id"]
          },
        ]
      }
      tennis_players: {
        Row: {
          birth_date: string | null
          country_code: string | null
          created_at: string
          handedness: string | null
          id: string
          name: string
          provider_player_id: string | null
          tour: string
          updated_at: string
        }
        Insert: {
          birth_date?: string | null
          country_code?: string | null
          created_at?: string
          handedness?: string | null
          id?: string
          name: string
          provider_player_id?: string | null
          tour: string
          updated_at?: string
        }
        Update: {
          birth_date?: string | null
          country_code?: string | null
          created_at?: string
          handedness?: string | null
          id?: string
          name?: string
          provider_player_id?: string | null
          tour?: string
          updated_at?: string
        }
        Relationships: []
      }
      tennis_predictions: {
        Row: {
          created_at: string
          edge_a: number | null
          edge_b: number | null
          ev_a: number | null
          ev_b: number | null
          fair_odds_a: number
          fair_odds_b: number
          feature_version: string
          id: string
          market_probability_a: number | null
          market_probability_b: number | null
          match_id: string
          model_run_id: string | null
          odds_snapshot_id: number | null
          probability_a: number
          probability_b: number
          recommendation: string
          stake_units: number
          tier: string
          uncertainty: number
        }
        Insert: {
          created_at?: string
          edge_a?: number | null
          edge_b?: number | null
          ev_a?: number | null
          ev_b?: number | null
          fair_odds_a: number
          fair_odds_b: number
          feature_version: string
          id?: string
          market_probability_a?: number | null
          market_probability_b?: number | null
          match_id: string
          model_run_id?: string | null
          odds_snapshot_id?: number | null
          probability_a: number
          probability_b: number
          recommendation: string
          stake_units?: number
          tier: string
          uncertainty: number
        }
        Update: {
          created_at?: string
          edge_a?: number | null
          edge_b?: number | null
          ev_a?: number | null
          ev_b?: number | null
          fair_odds_a?: number
          fair_odds_b?: number
          feature_version?: string
          id?: string
          market_probability_a?: number | null
          market_probability_b?: number | null
          match_id?: string
          model_run_id?: string | null
          odds_snapshot_id?: number | null
          probability_a?: number
          probability_b?: number
          recommendation?: string
          stake_units?: number
          tier?: string
          uncertainty?: number
        }
        Relationships: [
          {
            foreignKeyName: "tennis_predictions_match_id_fkey"
            columns: ["match_id"]
            isOneToOne: false
            referencedRelation: "tennis_matches"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tennis_predictions_model_run_id_fkey"
            columns: ["model_run_id"]
            isOneToOne: false
            referencedRelation: "tennis_model_runs"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tennis_predictions_odds_snapshot_id_fkey"
            columns: ["odds_snapshot_id"]
            isOneToOne: false
            referencedRelation: "tennis_odds_snapshots"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends (DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends (DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends (DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends (DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never) = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends (PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never) = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
