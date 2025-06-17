from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

class AlertPanel(QWidget):
    """A panel for displaying anomaly events and AI reasoning."""
    feedback_provided = pyqtSignal(dict, str)
    analysis_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active_event_data = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Initializes the user interface of the alert panel."""
        self.setStyleSheet("""
            #AlertContentFrame {
                background-color: #3c3c3c;
                border-radius: 8px;
            }
            QListWidget, QFrame#InfoBox {
                background-color: transparent;
                border: 1px solid #444;
                border-radius: 5px;
            }
            QLabel { font-size: 14px; color: #e0e0e0; }
            QLabel#TitleLabel { font-size: 18px; font-weight: bold; padding-bottom: 10px; color: #ffffff; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #444; }
            QListWidget::item:selected { background-color: #06C755; color: #ffffff; }
        """)

        # Main layout for the entire widget
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Content frame that has the styled background and border
        content_frame = QFrame()
        content_frame.setObjectName("AlertContentFrame")
        main_layout = QVBoxLayout(content_frame)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        outer_layout.addWidget(content_frame)

        title_label = QLabel("Anomaly Detection & AI Reasoning")
        title_label.setObjectName("TitleLabel")
        main_layout.addWidget(title_label)

        main_layout.addWidget(QLabel("Detected Anomalies:"))
        self.event_list_widget = QListWidget()
        self.event_list_widget.itemClicked.connect(self._on_event_clicked)
        main_layout.addWidget(self.event_list_widget, 1)

        main_layout.addWidget(self._create_separator())
        main_layout.addWidget(QLabel("AI Reasoning Process:"))

        self.llm_label_box = self._create_info_box("LLM Generated Constraint")
        self.vlm_label_box = self._create_info_box("VLM Analysis & Decision")
        main_layout.addWidget(self.llm_label_box)
        main_layout.addWidget(self.vlm_label_box)
        main_layout.addStretch(1)

        self._init_feedback_buttons(main_layout)

    def _create_info_box(self, title: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("InfoBox")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("font-weight: bold; color: #4CC764; border: none; background: none;")
        info_label = QLabel("No event selected.")
        info_label.setWordWrap(True)
        info_label.setObjectName("info_label")
        info_label.setStyleSheet("border: none; background: none;")
        layout.addWidget(title_label)
        layout.addWidget(info_label)
        return frame

    def _init_feedback_buttons(self, parent_layout: QVBoxLayout) -> None:
        feedback_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("✔️ Confirm Cheating")
        self.false_pos_btn = QPushButton("❌ False Positive")
        self.confirm_btn.setStyleSheet("background-color: #e74c3c;")
        self.false_pos_btn.setStyleSheet("background-color: #3498db;")
        self.confirm_btn.clicked.connect(lambda: self.emit_feedback("confirmed_cheating"))
        self.false_pos_btn.clicked.connect(lambda: self.emit_feedback("false_positive"))
        feedback_layout.addWidget(self.confirm_btn)
        feedback_layout.addWidget(self.false_pos_btn)
        parent_layout.addLayout(feedback_layout)

    def add_event(self, event_data: dict) -> None:
        item_text = f"[{event_data['timestamp']:.2f}s] {event_data.get('type', 'Event').replace('_', ' ').title()}"
        list_item = QListWidgetItem(item_text)
        list_item.setData(Qt.ItemDataRole.UserRole, event_data)
        self.event_list_widget.addItem(list_item)
    
    def _on_event_clicked(self, item: QListWidgetItem) -> None:
        """Handles user clicking an event, setting UI to loading and requesting analysis."""
        event_data = item.data(Qt.ItemDataRole.UserRole)
        self.active_event_data = event_data
        
        llm_info = self.llm_label_box.findChild(QLabel, "info_label")
        vlm_info = self.vlm_label_box.findChild(QLabel, "info_label")

        if llm_info: llm_info.setText("<i>Requesting AI analysis...</i>")
        if vlm_info: vlm_info.setText("<i>Waiting for LLM constraints...</i>")
        
        self.analysis_requested.emit(event_data)

    def update_llm_constraint(self, constraint: str) -> None:
        if self.active_event_data:
            info_label = self.llm_label_box.findChild(QLabel, "info_label")
            if info_label: info_label.setText(constraint)

    def update_vlm_result(self, data: dict) -> None:
        event_data = data.get('event_data', {})
        if self.active_event_data and self.active_event_data.get('event_id') == event_data.get('event_id'):
            vlm_result = data.get('vlm_result', {})
            decision = vlm_result.get('decision', 'N/A')
            explanation = vlm_result.get('explanation', '')
            info_text = f"<b>{decision}</b><br>{explanation}"
            info_label = self.vlm_label_box.findChild(QLabel, "info_label")
            if info_label:
                info_label.setText(info_text)
                if "Confirmed" in decision: info_label.setStyleSheet("color: #e74c3c; border: none; background: none;")
                elif "Not Cheating" in decision: info_label.setStyleSheet("color: #4CC764; border: none; background: none;")
                else: info_label.setStyleSheet("color: #e0e0e0; border: none; background: none;")
            self.active_event_data['vlm_decision_text'] = decision
            self.active_event_data['vlm_explanation_text'] = explanation

    def emit_feedback(self, feedback_type: str) -> None:
        if self.active_event_data:
            self.feedback_provided.emit(self.active_event_data, feedback_type)
            self.confirm_btn.setEnabled(False)
            self.false_pos_btn.setEnabled(False)

    def clear_all_events(self) -> None:
        self.event_list_widget.clear()
        llm_info = self.llm_label_box.findChild(QLabel, "info_label")
        vlm_info = self.vlm_label_box.findChild(QLabel, "info_label")
        if llm_info: llm_info.setText("No event selected.")
        if vlm_info: vlm_info.setText("No event selected.")
        self.active_event_data = None

    def _create_separator(self) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #444;")
        return separator

