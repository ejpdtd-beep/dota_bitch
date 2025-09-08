package com.skadistats.clarity.fantasy;

import com.skadistats.clarity.Clarity;
import com.skadistats.clarity.model.s2.S2CombatLogEntry;
import com.skadistats.clarity.processor.reader.OnMessage;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class FantasyStatsProcessor {

    private Map<Integer, Map<String, Double>> playerStats = new HashMap<>();

    @OnMessage(S2CombatLogEntry.class)
    public void onCombatLog(S2CombatLogEntry entry) {
        int pid = entry.getPlayerId();
        playerStats.putIfAbsent(pid, new HashMap<>());
        Map<String, Double> stats = playerStats.get(pid);

        // Example: parse combat log event types and count
        stats.put("kills", stats.getOrDefault("kills", 0.0) + entry.getKills());
        stats.put("deaths", stats.getOrDefault("deaths", 0.0) + entry.getDeaths());
        stats.put("last_hits", stats.getOrDefault("last_hits", 0.0) + entry.getLastHits());
        stats.put("gpm", stats.getOrDefault("gpm", 0.0) + entry.getGoldPerMin());
        stats.put("madstones", stats.getOrDefault("madstones", 0.0) + entry.getMadstonesCollected());
        stats.put("tower_kills", stats.getOrDefault("tower_kills", 0.0) + entry.getTowerKills());
        stats.put("wards", stats.getOrDefault("wards", 0.0) + entry.getWardsPlaced());
        stats.put("camps", stats.getOrDefault("camps", 0.0) + entry.getCampsStacked());
        stats.put("runes", stats.getOrDefault("runes", 0.0) + entry.getRunesGrabbed());
        stats.put("watchers", stats.getOrDefault("watchers", 0.0) + entry.getWatchersTaken());
        stats.put("lotuses", stats.getOrDefault("lotuses", 0.0) + entry.getLotusesGrabbed());
        stats.put("roshan", stats.getOrDefault("roshan", 0.0) + entry.getRoshanKills());
        stats.put("teamfight", stats.getOrDefault("teamfight", 0.0) + entry.getTeamfightPoints());
        stats.put("stuns", stats.getOrDefault("stuns", 0.0) + entry.getStunDuration());
        stats.put("tormentor", stats.getOrDefault("tormentor", 0.0) + entry.getTormentorKills());
        stats.put("courier", stats.getOrDefault("courier", 0.0) + entry.getCourierKills());
        stats.put("first_blood", stats.getOrDefault("first_blood", 0.0) + entry.getFirstBlood());
        stats.put("smokes", stats.getOrDefault("smokes", 0.0) + entry.getSmokesUsed());
    }

    public void exportCSV(String filename) throws IOException {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("player_id,kills,deaths,last_hits,gpm,madstones,tower_kills,wards,camps,runes,watchers,lotuses,roshan,teamfight,stuns,tormentor,courier,first_blood,smokes\n");
            for (Map.Entry<Integer, Map<String, Double>> entry : playerStats.entrySet()) {
                Integer pid = entry.getKey();
                Map<String, Double> s = entry.getValue();
                writer.write(pid + "," + s.getOrDefault("kills", 0.0) + "," +
                        s.getOrDefault("deaths", 0.0) + "," +
                        s.getOrDefault("last_hits", 0.0) + "," +
                        s.getOrDefault("gpm", 0.0) + "," +
                        s.getOrDefault("madstones", 0.0) + "," +
                        s.getOrDefault("tower_kills", 0.0) + "," +
                        s.getOrDefault("wards", 0.0) + "," +
                        s.getOrDefault("camps", 0.0) + "," +
                        s.getOrDefault("runes", 0.0) + "," +
                        s.getOrDefault("watchers", 0.0) + "," +
                        s.getOrDefault("lotuses", 0.0) + "," +
                        s.getOrDefault("roshan", 0.0) + "," +
                        s.getOrDefault("teamfight", 0.0) + "," +
                        s.getOrDefault("stuns", 0.0) + "," +
                        s.getOrDefault("tormentor", 0.0) + "," +
                        s.getOrDefault("courier", 0.0) + "," +
                        s.getOrDefault("first_blood", 0.0) + "," +
                        s.getOrDefault("smokes", 0.0) + "\n");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Clarity clarity = new Clarity();
        clarity.open(args[0]); // path to replay.dem
        FantasyStatsProcessor processor = new FantasyStatsProcessor();
        clarity.read(processor);
        processor.exportCSV(args[1]); // output CSV
    }
}
