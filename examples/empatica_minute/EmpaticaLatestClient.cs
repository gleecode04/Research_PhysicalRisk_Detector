using System;
using System.Collections;
using System.IO;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;

public class EmpaticaLatestClient : MonoBehaviour
{
    // =========================
    // Config (JSON) additions
    // =========================
    [Header("Runtime Config (JSON file on device/PC)")]
    [Tooltip("If true, loads host IP/ports/paths/poll interval from a JSON file in persistentDataPath.")]
    public bool useRuntimeConfig = true;

    [Tooltip("Folder name under Application.persistentDataPath to store the JSON config.")]
    public string configFolderName = "EmpaticaConfig";

    [Tooltip("JSON config filename.")]
    public string configFileName = "empatica_config.json";

    [Serializable]
    private class EmpaticaConfig
    {
        // Biomarker server
        public string hostIp;
        public int port;
        public string path;

        // Prediction server
        public string predictHostIp;
        public int predictPort;
        public string predictPath;

        // Poll interval
        public float pollIntervalSeconds;

        // Optional override
        public string participantIdOverride;

        // Optional logging preferences (kept minimal; you already have these fields)
        public bool newCsvPerScene;
    }

    private string _configDir;
    private string _configPath;

    // =========================
    // Existing fields
    // =========================
    [Header("Biomarker Server (Laptop running avro_stream_poller.py)")]
    public string hostIp = "10.182.63.242";
    public int port = 7000;
    public string path = "/latest";
    public float pollIntervalSeconds = 5f;
    public string LatestUrl => $"http://{hostIp}:{port}{path}";

    [Header("Prediction Server (Laptop running prediction_server.py)")]
    public string predictHostIp = "10.182.63.242";
    public int predictPort = 5000;
    public string predictPath = "/predict";
    public string PredictUrl => $"http://{predictHostIp}:{predictPort}{predictPath}";

    [Header("Latest Biomarkers")]
    public float eda;
    public float temp;
    public float pulse_rate;
    public string eda_time;
    public string temp_time;
    public string pulse_time;
    public string participant_id;

    public bool hasData;
    public bool isNewMinute;
    public string lastProcessedMinute;

    [Header("ML Predictions (last known)")]
    public int stress;
    public string stress_label;
    public int cogload;
    public string cogload_label;
    public bool hasPrediction;

    [Header("UI (TextMeshPro)")]
    public TMP_Text predictionText;
    public bool showBioValues = true;

    [Header("High Stress UI (Single Canvas)")]
    public GameObject highStressCanvas;
    public string highStressLabel = "high";
    public int highStressThreshold = 2;
    public float highStressCooldownSeconds = 60f;
    private float lastHighStressShownTime = -999f;

    [Header("High Cogload UI (Single Canvas)")]
    public GameObject highCogloadCanvas;
    public string highCogloadLabel = "high";
    public int highCogloadThreshold = 2;
    public float highCogloadCooldownSeconds = 60f;
    private float lastHighCogloadShownTime = -999f;

    [Header("Logging (Saved on Device)")]
    public bool enableLogging = true;

    [Tooltip("Leave empty to auto-generate a new folder each time the APP starts.")]
    public string sessionId = "";

    public string participantIdOverride = "";
    public bool flushEveryWrite = true;

    [Tooltip("If true, a NEW CSV is started whenever the active scene changes (same session folder).")]
    public bool newCsvPerScene = false;

    private string _logDir;
    private string _csvPath;
    private StreamWriter _writer;
    private DateTime _sessionStartUtc;
    private string _sidUsed = "";
    private string _activeSceneName = "";

    // Keep latest raw JSONs (optional, useful for debugging)
    private string _latestRawJson = "";
    private string _lastPredictReqJson = "";
    private string _lastPredictRespJson = "";

    [Serializable]
    private class LatestPayload
    {
        public float eda;
        public float temp;
        public float pulse_rate;
        public string eda_time;
        public string temp_time;
        public string pulse_time;
        public string participant_id;
        public string note;
        public string updated_at;
    }

    [Serializable]
    private class PredictRequest
    {
        public float eda;
        public float bvp;
        public float temp;
    }

    [Serializable]
    private class PredictResponse
    {
        public int stress;
        public string stress_label;
        public int cogload;
        public string cogload_label;
    }

    private void Start()
    {
        if (highStressCanvas != null) highStressCanvas.SetActive(false);
        if (highCogloadCanvas != null) highCogloadCanvas.SetActive(false);

        // Load or create runtime config FIRST (so IP/ports/paths are correct before polling)
        InitAndLoadRuntimeConfig();

        _sessionStartUtc = DateTime.UtcNow;
        _activeSceneName = CurrentSceneName();

        InitSessionFolder();
        OpenCsvForCurrentScene(); // opens writer + header + session_start row

        if (newCsvPerScene)
            SceneManager.activeSceneChanged += OnActiveSceneChanged;

        UpdateUI("Waiting for data...");
        StartCoroutine(PollLoop());
    }

    private void OnApplicationQuit()
    {
        WriteSessionRow("session_stop");
        CloseLogging();

        if (newCsvPerScene)
            SceneManager.activeSceneChanged -= OnActiveSceneChanged;
    }

    private void OnDestroy()
    {
        if (_writer != null)
        {
            WriteSessionRow("session_stop");
            CloseLogging();
        }

        if (newCsvPerScene)
            SceneManager.activeSceneChanged -= OnActiveSceneChanged;
    }

    private void OnActiveSceneChanged(Scene oldScene, Scene newScene)
    {
        _activeSceneName = newScene.name;

        if (!newCsvPerScene) return;

        CloseLogging();
        OpenCsvForCurrentScene();
    }

    // =========================
    // Runtime Config (JSON)
    // =========================
    private void InitAndLoadRuntimeConfig()
    {
        _configDir = Path.Combine(UnityEngine.Application.persistentDataPath, configFolderName);
        _configPath = Path.Combine(_configDir, configFileName);

        // Always ensure directory exists
        Directory.CreateDirectory(_configDir);

        Debug.Log($"[EmpaticaLatestClient] Runtime config path: {_configPath}");

        if (!useRuntimeConfig) return;

        try
        {
            if (!File.Exists(_configPath))
            {
                // Create a default config from current Inspector values
                var cfg = new EmpaticaConfig
                {
                    hostIp = hostIp,
                    port = port,
                    path = path,
                    predictHostIp = predictHostIp,
                    predictPort = predictPort,
                    predictPath = predictPath,
                    pollIntervalSeconds = pollIntervalSeconds,
                    participantIdOverride = participantIdOverride,
                    newCsvPerScene = newCsvPerScene
                };

                string json = JsonUtility.ToJson(cfg, prettyPrint: true);
                File.WriteAllText(_configPath, json, Encoding.UTF8);

                Debug.Log("[EmpaticaLatestClient] Config file created (edit this file to change IP without rebuilding).");
                // Still continue using the current Inspector values (already set)
                return;
            }

            // Load existing config and apply
            string loadedJson = File.ReadAllText(_configPath, Encoding.UTF8);
            var loaded = JsonUtility.FromJson<EmpaticaConfig>(loadedJson);

            if (loaded != null)
            {
                // Only apply if values are present / valid
                if (!string.IsNullOrWhiteSpace(loaded.hostIp)) hostIp = loaded.hostIp.Trim();
                if (loaded.port > 0) port = loaded.port;
                if (!string.IsNullOrWhiteSpace(loaded.path)) path = loaded.path.Trim();

                if (!string.IsNullOrWhiteSpace(loaded.predictHostIp)) predictHostIp = loaded.predictHostIp.Trim();
                if (loaded.predictPort > 0) predictPort = loaded.predictPort;
                if (!string.IsNullOrWhiteSpace(loaded.predictPath)) predictPath = loaded.predictPath.Trim();

                if (loaded.pollIntervalSeconds > 0.1f) pollIntervalSeconds = loaded.pollIntervalSeconds;

                if (!string.IsNullOrWhiteSpace(loaded.participantIdOverride))
                    participantIdOverride = loaded.participantIdOverride.Trim();

                // Optional: allow config to toggle this too
                newCsvPerScene = loaded.newCsvPerScene;

                Debug.Log($"[EmpaticaLatestClient] Config loaded. LatestUrl={LatestUrl} | PredictUrl={PredictUrl} | poll={pollIntervalSeconds}s");
            }
            else
            {
                Debug.LogWarning("[EmpaticaLatestClient] Config file exists but could not be parsed. Using Inspector values.");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EmpaticaLatestClient] Failed to load/create runtime config: {e.Message}. Using Inspector values.");
        }
    }

    // -------------------------
    // Logging (one row, append)
    // -------------------------
    private void InitSessionFolder()
    {
        if (!enableLogging) return;

        _sidUsed = string.IsNullOrWhiteSpace(sessionId)
            ? DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
            : SanitizeFileName(sessionId.Trim());

        _logDir = Path.Combine(UnityEngine.Application.persistentDataPath, "EmpaticaLogs", _sidUsed);
        Directory.CreateDirectory(_logDir);

        Debug.Log($"[EmpaticaLatestClient] Session folder: {_logDir}");
    }

    private void OpenCsvForCurrentScene()
    {
        if (!enableLogging) return;

        try
        {
            string sceneName = _activeSceneName;
            string fileName = newCsvPerScene ? $"empatica_log_{SanitizeFileName(sceneName)}.csv" : "empatica_log.csv";

            _csvPath = Path.Combine(_logDir, fileName);

            bool isNew = !File.Exists(_csvPath);
            _writer = new StreamWriter(_csvPath, append: true, encoding: new UTF8Encoding(false));

            if (isNew)
            {
                _writer.WriteLine(
                    "app_utc_iso,row_type,scene_name,participant_id," +
                    "eda,temp,pulse_rate," +
                    "eda_time,temp_time,pulse_time,source_updated_at," +
                    "stress,stress_label,cogload,cogload_label," +
                    "latest_raw_json,last_predict_request_json,last_predict_response_json"
                );
                if (flushEveryWrite) _writer.Flush();
            }

            Debug.Log($"[EmpaticaLatestClient] Logging file: {_csvPath}");

            WriteSessionRow("session_start");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EmpaticaLatestClient] Failed to open CSV: {e.Message}");
            enableLogging = false;
        }
    }

    private void CloseLogging()
    {
        try
        {
            _writer?.Flush();
            _writer?.Close();
            _writer = null;
        }
        catch { }
    }

    private static string SanitizeFileName(string s)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            s = s.Replace(c, '_');
        return s;
    }

    private static string Csv(string value)
    {
        if (value == null) return "";
        bool mustQuote = value.Contains(",") || value.Contains("\"") || value.Contains("\n") || value.Contains("\r");
        if (value.Contains("\"")) value = value.Replace("\"", "\"\"");
        return mustQuote ? $"\"{value}\"" : value;
    }

    private string CurrentSceneName()
    {
        try { return SceneManager.GetActiveScene().name; }
        catch { return "unknown_scene"; }
    }

    private string EffectiveParticipantId()
    {
        if (!string.IsNullOrWhiteSpace(participantIdOverride))
            return participantIdOverride.Trim();
        return participant_id ?? "";
    }

    // Writes ONE row snapshot (biomarkers + last known prediction) every poll interval
    private void WriteSnapshotRow(string rowType)
    {
        if (!enableLogging || _writer == null) return;

        string now = DateTime.UtcNow.ToString("o");
        string sceneName = CurrentSceneName();
        string pid = EffectiveParticipantId();

        string line =
            $"{Csv(now)}," +
            $"{Csv(rowType)}," +
            $"{Csv(sceneName)}," +
            $"{Csv(pid)}," +
            $"{(hasData ? eda.ToString("G") : "")}," +
            $"{(hasData ? temp.ToString("G") : "")}," +
            $"{(hasData ? pulse_rate.ToString("G") : "")}," +
            $"{Csv(hasData ? eda_time : "")}," +
            $"{Csv(hasData ? temp_time : "")}," +
            $"{Csv(hasData ? pulse_time : "")}," +
            $"{Csv("")}," +
            $"{(hasPrediction ? stress.ToString() : "")}," +
            $"{Csv(hasPrediction ? stress_label : "")}," +
            $"{(hasPrediction ? cogload.ToString() : "")}," +
            $"{Csv(hasPrediction ? cogload_label : "")}," +
            $"{Csv(_latestRawJson)}," +
            $"{Csv(_lastPredictReqJson)}," +
            $"{Csv(_lastPredictRespJson)}";

        _writer.WriteLine(line);
        if (flushEveryWrite) _writer.Flush();
    }

    private void WriteSessionRow(string type)
    {
        if (!enableLogging || _writer == null) return;

        string now = DateTime.UtcNow.ToString("o");
        string sceneName = CurrentSceneName();
        string pid = EffectiveParticipantId();

        string line =
            $"{Csv(now)}," +
            $"{Csv(type)}," +
            $"{Csv(sceneName)}," +
            $"{Csv(pid)}," +
            ",,," +
            ",,," +
            "," +
            ",,,," +
            ",,,";

        _writer.WriteLine(line);
        if (flushEveryWrite) _writer.Flush();
    }

    // -------------------------
    // Polling
    // -------------------------
    private IEnumerator PollLoop()
    {
        while (true)
        {
            yield return FetchLatest();

            if (hasData && isNewMinute)
                yield return SendForPrediction();

            // ONE ROW every poll interval: biomarker + last known prediction
            WriteSnapshotRow("sample");

            yield return new WaitForSeconds(pollIntervalSeconds);
        }
    }

    private IEnumerator FetchLatest()
    {
        using (var req = UnityWebRequest.Get(LatestUrl))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                hasData = false;
                isNewMinute = false;
                hasPrediction = false;

                Debug.LogWarning($"[EmpaticaLatestClient] GET /latest failed: {req.error} | URL={LatestUrl}");
                UpdateUI($"BIO error:\n{req.error}");

                if (highStressCanvas != null) highStressCanvas.SetActive(false);
                if (highCogloadCanvas != null) highCogloadCanvas.SetActive(false);

                yield break;
            }

            var json = req.downloadHandler.text;

            if (string.IsNullOrWhiteSpace(json) || json.Trim() == "{}")
            {
                hasData = false;
                isNewMinute = false;
                yield break;
            }

            try
            {
                _latestRawJson = json;

                var payload = JsonUtility.FromJson<LatestPayload>(json);

                eda = payload.eda;
                temp = payload.temp;
                pulse_rate = payload.pulse_rate;

                eda_time = payload.eda_time;
                temp_time = payload.temp_time;
                pulse_time = payload.pulse_time;

                participant_id = payload.participant_id;

                if (!string.IsNullOrEmpty(eda_time) && eda_time != lastProcessedMinute)
                {
                    isNewMinute = true;
                    lastProcessedMinute = eda_time;
                }
                else
                {
                    isNewMinute = false;
                }

                hasData = true;
                UpdateUI(hasPrediction ? BuildPredictionString() : BuildStatusString());
            }
            catch (Exception e)
            {
                hasData = false;
                isNewMinute = false;

                Debug.LogWarning($"[EmpaticaLatestClient] JSON parse error (/latest): {e.Message}\nRaw: {json}\nURL={LatestUrl}");
                UpdateUI("BIO parse error.");
            }
        }
    }

    private IEnumerator SendForPrediction()
    {
        float bvp_proxy = pulse_rate;

        var reqObj = new PredictRequest { eda = eda, bvp = bvp_proxy, temp = temp };
        string bodyJson = JsonUtility.ToJson(reqObj);
        byte[] bodyRaw = Encoding.UTF8.GetBytes(bodyJson);

        using (var req = new UnityWebRequest(PredictUrl, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = 10;

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                hasPrediction = false;
                Debug.LogWarning($"[EmpaticaLatestClient] POST /predict failed: {req.error} | URL={PredictUrl} | Body={bodyJson}");
                UpdateUI($"Prediction error:\n{req.error}");
                yield break;
            }

            var json = req.downloadHandler.text;

            try
            {
                _lastPredictReqJson = bodyJson;
                _lastPredictRespJson = json;

                var pred = JsonUtility.FromJson<PredictResponse>(json);

                stress = pred.stress;
                stress_label = pred.stress_label;
                cogload = pred.cogload;
                cogload_label = pred.cogload_label;
                hasPrediction = true;

                HandleHighStressCanvas();
                HandleHighCogloadCanvas();

                UpdateUI(BuildPredictionString());
            }
            catch (Exception e)
            {
                hasPrediction = false;
                Debug.LogWarning($"[EmpaticaLatestClient] JSON parse error (/predict): {e.Message}\nRaw: {json}\nURL={PredictUrl}");
                UpdateUI("Prediction parse error.");
            }
        }
    }

    private void HandleHighStressCanvas()
    {
        if (highStressCanvas == null) return;

        bool high = IsHighStress();

        if (high)
        {
            if (!highStressCanvas.activeSelf && (Time.time - lastHighStressShownTime) >= highStressCooldownSeconds)
            {
                lastHighStressShownTime = Time.time;
                highStressCanvas.SetActive(true);
            }
        }
        else if (highStressCanvas.activeSelf)
        {
            highStressCanvas.SetActive(false);
        }
    }

    private void HandleHighCogloadCanvas()
    {
        if (highCogloadCanvas == null) return;

        bool high = IsHighCogload();

        if (high)
        {
            if (!highCogloadCanvas.activeSelf && (Time.time - lastHighCogloadShownTime) >= highCogloadCooldownSeconds)
            {
                lastHighCogloadShownTime = Time.time;
                highCogloadCanvas.SetActive(true);
            }
        }
        else if (highCogloadCanvas.activeSelf)
        {
            highCogloadCanvas.SetActive(false);
        }
    }

    private bool IsHighStress()
    {
        if (!string.IsNullOrWhiteSpace(stress_label))
            return stress_label.Trim().Equals(highStressLabel, StringComparison.OrdinalIgnoreCase);
        return stress >= highStressThreshold;
    }

    private bool IsHighCogload()
    {
        if (!string.IsNullOrWhiteSpace(cogload_label))
            return cogload_label.Trim().Equals(highCogloadLabel, StringComparison.OrdinalIgnoreCase);
        return cogload >= highCogloadThreshold;
    }

    public void DismissHighStressCanvas()
    {
        if (highStressCanvas != null) highStressCanvas.SetActive(false);
    }

    public void DismissHighCogloadCanvas()
    {
        if (highCogloadCanvas != null) highCogloadCanvas.SetActive(false);
    }

    private string BuildPredictionString()
    {
        string msg = $"Stress: {stress_label}\nCogload: {cogload_label}";

        if (showBioValues && hasData)
        {
            msg += $"\n\nEDA: {eda:G}\nTEMP: {temp:G}\nBVP: {pulse_rate:G}";
        }
        return msg;
    }

    private string BuildStatusString()
    {
        if (!hasData) return "Waiting for data...";

        string msg = "Waiting for prediction...";
        if (showBioValues)
        {
            msg += $"\n\nEDA: {eda:G}\nTEMP: {temp:G}\nBVP: {pulse_rate:G}";
        }
        return msg;
    }

    private void UpdateUI(string message)
    {
        if (predictionText != null) predictionText.text = message;
    }
}